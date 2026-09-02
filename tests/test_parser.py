"""Tests for the safe expression parser."""
import numpy as np
import pytest

from math_engine.parser import ExpressionError, parse_expression, to_latex


class TestValidExpressions:
    def test_sine(self):
        p = parse_expression("sin(x)", ["x"])
        assert np.allclose(p.evaluate(np.array([0.0])), [0.0])

    def test_polynomial(self):
        p = parse_expression("x^2 + 2*x + 1", ["x"])
        assert np.allclose(p.evaluate(np.array([1.0])), [4.0])

    def test_caret_power(self):
        p = parse_expression("x^3", ["x"])
        assert np.allclose(p.evaluate(np.array([2.0])), [8.0])

    def test_constants(self):
        p = parse_expression("pi", ["x"])
        assert np.allclose(p.evaluate(np.array([0.0])), [np.pi])

    def test_two_variables(self):
        p = parse_expression("x^2 + y^2 - 1", ["x", "y"])
        assert np.allclose(p.evaluate(np.array([1.0]), np.array([0.0])), [0.0])

    def test_theta_variable(self):
        p = parse_expression("1 + 0.5*cos(5*theta)", ["theta"])
        assert np.allclose(p.evaluate(np.array([0.0])), [1.5])

    def test_implicit_multiplication(self):
        p = parse_expression("2x", ["x"])
        assert np.allclose(p.evaluate(np.array([3.0])), [6.0])

    def test_constant_expression_broadcasts(self):
        p = parse_expression("3", ["x"])
        out = p.evaluate(np.linspace(0, 1, 7))
        assert out.shape == (7,)
        assert np.allclose(out, 3.0)


class TestInvalidExpressions:
    def test_undefined_variable(self):
        with pytest.raises(ExpressionError, match="Undefined variable"):
            parse_expression("sin(q)", ["x"])

    def test_theta_not_allowed_in_cartesian(self):
        with pytest.raises(ExpressionError, match="Undefined variable"):
            parse_expression("cos(theta)", ["x"])

    def test_bad_syntax(self):
        with pytest.raises(ExpressionError):
            parse_expression("sin(x", ["x"])

    def test_empty(self):
        with pytest.raises(ExpressionError):
            parse_expression("", ["x"])

    def test_dunder_blocked(self):
        with pytest.raises(ExpressionError, match="forbidden"):
            parse_expression("__import__('os')", ["x"])

    def test_import_blocked(self):
        with pytest.raises(ExpressionError, match="forbidden"):
            parse_expression("import os", ["x"])

    def test_lambda_blocked(self):
        with pytest.raises(ExpressionError, match="forbidden"):
            parse_expression("lambda x: x", ["x"])

    def test_quotes_blocked(self):
        with pytest.raises(ExpressionError):
            parse_expression("open('/etc/passwd')", ["x"])

    def test_too_long(self):
        with pytest.raises(ExpressionError):
            parse_expression("x+" * 500 + "x", ["x"])


class TestNaNInfinityHandling:
    def test_division_pole_gives_nonfinite(self):
        p = parse_expression("1/x", ["x"])
        out = p.evaluate(np.array([0.0, 1.0]))
        assert not np.isfinite(out[0])
        assert np.isclose(out[1], 1.0)

    def test_sqrt_negative_gives_nan(self):
        p = parse_expression("sqrt(x)", ["x"])
        out = p.evaluate(np.array([-1.0, 4.0]))
        assert np.isnan(out[0])
        assert np.isclose(out[1], 2.0)

    def test_log_nonpositive_gives_nonfinite(self):
        p = parse_expression("log(x)", ["x"])
        out = p.evaluate(np.array([-1.0, 0.0, np.e]))
        assert not np.isfinite(out[0])
        assert not np.isfinite(out[1])
        assert np.isclose(out[2], 1.0)


def test_to_latex():
    assert "sin" in to_latex("sin(x)", ["x"])
    # Falls back to raw text on invalid input
    assert to_latex("sin(", ["x"]) == "sin("
