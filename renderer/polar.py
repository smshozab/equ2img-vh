"""Polar r = f(theta) renderer (converted to cartesian)."""
from __future__ import annotations

import numpy as np

from math_engine.parser import parse_expression
from math_engine.sampler import sample_polar
from schema.scene import PolarObject


def render_polar(ax, obj: PolarObject) -> tuple[list[np.ndarray], float]:
    parsed = parse_expression(obj.equation, ["theta"])
    curve = sample_polar(
        parsed,
        (obj.range[0], obj.range[1]),
        obj.samples,
        center=(obj.center[0], obj.center[1]),
    )
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
