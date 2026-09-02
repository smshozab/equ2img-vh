"""Numerical sampling utilities: discontinuity handling, grid sampling."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from math_engine.parser import ParsedExpression


@dataclass
class SampledCurve:
    """A curve split into continuous segments (each an (N,2) array)."""

    segments: list[np.ndarray]
    total_points: int
    finite_fraction: float


def _split_on_breaks(x: np.ndarray, y: np.ndarray, jump_factor: float = 50.0) -> list[np.ndarray]:
    """Split x/y arrays into continuous polyline segments.

    Breaks at non-finite values and at jumps much larger than the median
    step (catches poles like tan(x)).
    """
    pts = np.column_stack([x, y])
    finite = np.isfinite(pts).all(axis=1)

    # Distances between consecutive finite points for jump detection
    d = np.hypot(np.diff(x), np.diff(y))
    with np.errstate(invalid="ignore"):
        finite_d = d[np.isfinite(d)]
    if finite_d.size:
        med = float(np.median(finite_d))
        threshold = max(med * jump_factor, 1e-12)
    else:
        threshold = np.inf

    segments: list[np.ndarray] = []
    current: list[np.ndarray] = []
    for i in range(len(pts)):
        if not finite[i]:
            if len(current) >= 2:
                segments.append(np.array(current))
            current = []
            continue
        if current and i > 0:
            step = d[i - 1]
            if not np.isfinite(step) or step > threshold:
                if len(current) >= 2:
                    segments.append(np.array(current))
                current = []
        current.append(pts[i])
    if len(current) >= 2:
        segments.append(np.array(current))
    return segments


def sample_cartesian(
    parsed: ParsedExpression, domain: tuple[float, float], samples: int
) -> SampledCurve:
    xs = np.linspace(domain[0], domain[1], samples)
    ys = parsed.evaluate(xs)
    finite_frac = float(np.isfinite(ys).mean()) if ys.size else 0.0
    segments = _split_on_breaks(xs, ys)
    return SampledCurve(segments=segments, total_points=samples, finite_fraction=finite_frac)


def sample_parametric(
    parsed_x: ParsedExpression,
    parsed_y: ParsedExpression,
    t_range: tuple[float, float],
    samples: int,
) -> SampledCurve:
    ts = np.linspace(t_range[0], t_range[1], samples)
    xs = parsed_x.evaluate(ts)
    ys = parsed_y.evaluate(ts)
    finite = np.isfinite(xs) & np.isfinite(ys)
    finite_frac = float(finite.mean()) if ts.size else 0.0
    segments = _split_on_breaks(xs, ys)
    return SampledCurve(segments=segments, total_points=samples, finite_fraction=finite_frac)


def sample_polar(
    parsed_r: ParsedExpression,
    theta_range: tuple[float, float],
    samples: int,
    center: tuple[float, float] = (0.0, 0.0),
) -> SampledCurve:
    ths = np.linspace(theta_range[0], theta_range[1], samples)
    rs = parsed_r.evaluate(ths)
    xs = rs * np.cos(ths) + center[0]
    ys = rs * np.sin(ths) + center[1]
    finite = np.isfinite(xs) & np.isfinite(ys)
    finite_frac = float(finite.mean()) if ths.size else 0.0
    segments = _split_on_breaks(xs, ys)
    return SampledCurve(segments=segments, total_points=samples, finite_fraction=finite_frac)


def sample_grid(
    parsed: ParsedExpression,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Sample F(x, y) on a grid. Returns (X, Y, Z, finite_fraction)."""
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(xs, ys)
    Z = parsed.evaluate(X, Y)
    finite_frac = float(np.isfinite(Z).mean()) if Z.size else 0.0
    return X, Y, Z, finite_frac
