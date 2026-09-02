"""Safety model: allowlisted functions/constants and expression limits.

LLM expressions are NEVER passed to eval(). They go through SymPy's
parse_expr with a restricted local dict and a token-level pre-check.
"""
from __future__ import annotations

import re

import sympy as sp

# --- Allowlisted mathematical functions -------------------------------------
ALLOWED_FUNCTIONS: dict[str, object] = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "atan2": sp.atan2,
    "sqrt": sp.sqrt,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "abs": sp.Abs,
    "Abs": sp.Abs,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "floor": sp.floor,
    "ceil": sp.ceiling,
    "ceiling": sp.ceiling,
    "sign": sp.sign,
    "min": sp.Min,
    "max": sp.Max,
    "Min": sp.Min,
    "Max": sp.Max,
}

ALLOWED_CONSTANTS: dict[str, object] = {
    "pi": sp.pi,
    "e": sp.E,
    "E": sp.E,
    "tau": 2 * sp.pi,
}

# Variables permitted per context are decided by the caller (parser.py).
ALLOWED_VARIABLE_NAMES = {"x", "y", "t", "theta", "u", "v", "s"}

MAX_EXPRESSION_LENGTH = 800
MAX_EXPRESSION_ATOMS = 400

# Reject anything that even smells like Python execution before parsing.
_FORBIDDEN_PATTERN = re.compile(
    r"(__|\bimport\b|\blambda\b|\bexec\b|\beval\b|\bopen\b|\bos\b|\bsys\b"
    r"|\bglobals\b|\blocals\b|\bgetattr\b|\bsetattr\b|[;@#]|:=|\"|')"
)

# Characters we allow in raw expressions.
_ALLOWED_CHARS = re.compile(r"^[0-9a-zA-Z_+\-*/^().,\s\[\]]+$")


class UnsafeExpressionError(ValueError):
    """Raised when an expression fails safety pre-checks."""


def precheck_expression(expr: str) -> str:
    """Validate raw expression text before it reaches the SymPy parser.

    Returns the stripped expression, or raises UnsafeExpressionError.
    """
    if not isinstance(expr, str):
        raise UnsafeExpressionError("Expression must be a string")
    expr = expr.strip()
    if not expr:
        raise UnsafeExpressionError("Expression is empty")
    if len(expr) > MAX_EXPRESSION_LENGTH:
        raise UnsafeExpressionError(
            f"Expression exceeds maximum length of {MAX_EXPRESSION_LENGTH} characters"
        )
    if _FORBIDDEN_PATTERN.search(expr):
        raise UnsafeExpressionError("Expression contains forbidden tokens")
    if not _ALLOWED_CHARS.match(expr):
        bad = sorted({c for c in expr if not _ALLOWED_CHARS.match(c)})
        raise UnsafeExpressionError(f"Expression contains unsupported characters: {bad}")
    return expr


def check_expression_size(parsed: sp.Expr) -> None:
    """Reject pathologically large symbolic expressions."""
    n_atoms = len(list(sp.preorder_traversal(parsed)))
    if n_atoms > MAX_EXPRESSION_ATOMS:
        raise UnsafeExpressionError(
            f"Expression too complex ({n_atoms} nodes > {MAX_EXPRESSION_ATOMS})"
        )
