"""Safe SymPy-based expression parsing and NumPy lambdification.

No eval() on raw strings: expressions pass safety prechecks, then are
parsed with sympy.parse_expr using a restricted local dict, then checked
for undefined symbols against an explicit variable allowlist.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from math_engine.safety import (
    ALLOWED_CONSTANTS,
    ALLOWED_FUNCTIONS,
    UnsafeExpressionError,
    check_expression_size,
    precheck_expression,
)

_TRANSFORMATIONS = standard_transformations + (
    convert_xor,  # allow ^ as power
    implicit_multiplication_application,  # allow 2x, 2 sin(x)
)


class ExpressionError(ValueError):
    """Raised when an expression cannot be parsed or validated."""


@dataclass(frozen=True)
class ParsedExpression:
    raw: str
    expr: sp.Expr
    variables: tuple[sp.Symbol, ...]
    func: Callable[..., np.ndarray]

    def evaluate(self, *arrays: np.ndarray) -> np.ndarray:
        """Evaluate numerically; always returns a float ndarray broadcast
        to the input shape, with non-finite values left as nan/inf."""
        with np.errstate(all="ignore"):
            result = self.func(*arrays)
        # Constant expressions return scalars - broadcast them.
        result = np.asarray(result, dtype=complex if np.iscomplexobj(result) else float)
        if arrays and result.shape != np.asarray(arrays[0]).shape:
            result = np.broadcast_to(result, np.asarray(arrays[0]).shape).copy()
        # Complex results (e.g. sqrt of negative) -> nan where imaginary
        if np.iscomplexobj(result):
            out = np.where(np.abs(result.imag) < 1e-12, result.real, np.nan)
            return out.astype(float)
        return result.astype(float)


def parse_expression(raw: str, allowed_vars: Sequence[str]) -> ParsedExpression:
    """Parse `raw` allowing only `allowed_vars` as free symbols.

    Raises ExpressionError with a human-readable message on failure.
    """
    try:
        cleaned = precheck_expression(raw)
    except UnsafeExpressionError as exc:
        raise ExpressionError(str(exc)) from exc

    local_dict: dict[str, object] = {}
    local_dict.update(ALLOWED_FUNCTIONS)
    local_dict.update(ALLOWED_CONSTANTS)
    symbols = {name: sp.Symbol(name, real=True) for name in allowed_vars}
    local_dict.update(symbols)

    try:
        expr = parse_expr(
            cleaned,
            local_dict=local_dict,
            transformations=_TRANSFORMATIONS,
            evaluate=True,
        )
    except Exception as exc:  # SymPy raises many exception types
        raise ExpressionError(f"Invalid syntax in expression '{raw}': {exc}") from exc

    if not isinstance(expr, sp.Basic):
        raise ExpressionError(f"Expression '{raw}' did not parse to a mathematical expression")

    try:
        check_expression_size(expr)
    except UnsafeExpressionError as exc:
        raise ExpressionError(str(exc)) from exc

    free = expr.free_symbols
    allowed_set = set(symbols.values())
    undefined = {s for s in free if s not in allowed_set}
    if undefined:
        names = ", ".join(sorted(str(s) for s in undefined))
        allowed_names = ", ".join(allowed_vars) if allowed_vars else "(none)"
        raise ExpressionError(
            f"Undefined variable(s) in expression '{raw}': {names}. "
            f"Allowed variables here: {allowed_names}"
        )

    ordered_vars = tuple(symbols[name] for name in allowed_vars)
    try:
        func = sp.lambdify(ordered_vars, expr, modules=["numpy"])
    except Exception as exc:
        raise ExpressionError(f"Cannot compile expression '{raw}': {exc}") from exc

    # Smoke test: evaluate on a tiny array to catch runtime-only problems.
    try:
        test_arrays = [np.array([0.5, 1.5]) for _ in ordered_vars]
        with np.errstate(all="ignore"):
            func(*test_arrays)
    except Exception as exc:
        raise ExpressionError(f"Expression '{raw}' fails numeric evaluation: {exc}") from exc

    return ParsedExpression(raw=raw, expr=expr, variables=ordered_vars, func=func)


def to_latex(raw: str, allowed_vars: Sequence[str]) -> str:
    """Best-effort LaTeX rendering of a raw expression string."""
    try:
        parsed = parse_expression(raw, allowed_vars)
        return sp.latex(parsed.expr)
    except ExpressionError:
        return raw
