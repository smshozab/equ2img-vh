"""Parametric x = f(t), y = g(t) renderer."""
from __future__ import annotations

import numpy as np

from math_engine.parser import parse_expression
from math_engine.sampler import sample_parametric
from schema.scene import ParametricObject


def render_parametric(ax, obj: ParametricObject) -> tuple[list[np.ndarray], float]:
    px = parse_expression(obj.x, [obj.parameter])
    py = parse_expression(obj.y, [obj.parameter])
    curve = sample_parametric(px, py, (obj.range[0], obj.range[1]), obj.samples)
    for seg in curve.segments:
        ax.plot(
            seg[:, 0], seg[:, 1],
            color=obj.style.color,
            linewidth=obj.style.line_width,
            linestyle=obj.style.mpl_linestyle,
            alpha=obj.style.alpha,
            zorder=obj.style.z_order,
            label=obj.label,
        )
        if obj.style.fill:
            ax.fill(
                seg[:, 0], seg[:, 1],
                color=obj.style.fill_color or obj.style.color,
                alpha=obj.style.alpha * 0.35,
                zorder=obj.style.z_order - 1,
            )
    return curve.segments, curve.finite_fraction
