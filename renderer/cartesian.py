"""Cartesian y = f(x) renderer."""
from __future__ import annotations

import numpy as np

from math_engine.parser import parse_expression
from math_engine.sampler import sample_cartesian
from schema.scene import CartesianObject


def render_cartesian(ax, obj: CartesianObject) -> tuple[list[np.ndarray], float]:
    parsed = parse_expression(obj.equation, ["x"])
    curve = sample_cartesian(parsed, (obj.domain[0], obj.domain[1]), obj.samples)
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
            ax.fill_between(
                seg[:, 0], seg[:, 1], 0,
                color=obj.style.fill_color or obj.style.color,
                alpha=obj.style.alpha * 0.35,
                zorder=obj.style.z_order - 1,
            )
    return curve.segments, curve.finite_fraction
