"""Implicit F(x,y)=0 curves (zero-contour extraction) and inequality regions."""
from __future__ import annotations

import numpy as np

from math_engine.parser import parse_expression
from math_engine.sampler import sample_grid
from schema.scene import ImplicitObject, RegionObject, Scene


def _object_domain(obj, scene: Scene) -> tuple[tuple[float, float], tuple[float, float]]:
    if obj.domain is not None:
        return (obj.domain.x[0], obj.domain.x[1]), (obj.domain.y[0], obj.domain.y[1])
    return (scene.bounds[0], scene.bounds[1]), (scene.bounds[2], scene.bounds[3])


def render_implicit(ax, obj: ImplicitObject, scene: Scene) -> tuple[list[np.ndarray], float]:
    parsed = parse_expression(obj.equation, ["x", "y"])
    x_range, y_range = _object_domain(obj, scene)
    X, Y, Z, finite_frac = sample_grid(parsed, x_range, y_range, obj.resolution)

    # Mask non-finite so contour extraction doesn't propagate garbage
    Zm = np.ma.masked_invalid(Z)
    cs = ax.contour(
        X, Y, Zm,
        levels=[0.0],
        colors=[obj.style.color],
        linewidths=[obj.style.line_width],
        linestyles=[obj.style.mpl_linestyle],
        alpha=obj.style.alpha,
        zorder=obj.style.z_order,
    )

    segments: list[np.ndarray] = []
    # Matplotlib >= 3.8: use allsegs (list per level of (N,2) arrays)
    for level_segs in cs.allsegs:
        for seg in level_segs:
            arr = np.asarray(seg)
            if len(arr) >= 2:
                segments.append(arr)

    if obj.style.fill:
        ax.contourf(
            X, Y, Zm,
            levels=[-1e30, 0.0],
            colors=[obj.style.fill_color or obj.style.color],
            alpha=obj.style.alpha * 0.35,
            zorder=obj.style.z_order - 1,
        )
    return segments, finite_frac


def render_region(ax, obj: RegionObject, scene: Scene) -> tuple[list[np.ndarray], float]:
    parsed = parse_expression(obj.equation, ["x", "y"])
    x_range, y_range = _object_domain(obj, scene)
    X, Y, Z, finite_frac = sample_grid(parsed, x_range, y_range, obj.resolution)

    Zm = np.ma.masked_invalid(Z)
    if obj.operator in ("<", "<="):
        levels = [-1e30, 0.0]
    else:
        levels = [0.0, 1e30]

    ax.contourf(
        X, Y, Zm,
        levels=levels,
        colors=[obj.style.fill_color or obj.style.color],
        alpha=obj.style.alpha * 0.6,
        zorder=obj.style.z_order - 1,
    )

    # Report the covered points as a pseudo-segment set for quality metrics
    if obj.operator in ("<", "<="):
        mask = Zm < 0
    else:
        mask = Zm > 0
    covered = int(np.ma.filled(mask, False).sum())
    if covered > 0:
        pts = np.column_stack([X[np.ma.filled(mask, False)], Y[np.ma.filled(mask, False)]])
        # subsample for reporting only
        step = max(1, len(pts) // 2000)
        return [pts[::step]], finite_frac
    return [], finite_frac
