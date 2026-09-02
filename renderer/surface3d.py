"""3D surface z = f(x, y) renderer (surface, wireframe, contour modes)."""
from __future__ import annotations

import numpy as np

from math_engine.parser import parse_expression
from math_engine.sampler import sample_grid
from schema.scene import Scene, Surface3DObject


def render_surface3d(ax, obj: Surface3DObject, scene: Scene, index: int):
    # Local import to avoid a circular dependency with renderer.base
    from renderer.base import ObjectReport

    try:
        parsed = parse_expression(obj.equation, ["x", "y"])
        if obj.domain is not None:
            x_range = (obj.domain.x[0], obj.domain.x[1])
            y_range = (obj.domain.y[0], obj.domain.y[1])
        else:
            x_range = (scene.bounds[0], scene.bounds[1])
            y_range = (scene.bounds[2], scene.bounds[3])

        X, Y, Z, finite_frac = sample_grid(parsed, x_range, y_range, obj.resolution)
        Z = np.where(np.isfinite(Z), Z, np.nan)

        if obj.mode == "wireframe":
            stride = max(1, obj.resolution // 40)
            ax.plot_wireframe(
                X, Y, Z,
                rstride=stride, cstride=stride,
                color=obj.style.color,
                linewidth=max(0.3, obj.style.line_width * 0.25),
                alpha=obj.style.alpha,
            )
        elif obj.mode == "contour":
            ax.contour3D(X, Y, Z, 40, cmap=obj.colormap, alpha=obj.style.alpha)
        else:
            ax.plot_surface(
                X, Y, Z,
                cmap=obj.colormap,
                alpha=obj.style.alpha,
                linewidth=0,
                antialiased=True,
            )

        points = int(np.isfinite(Z).sum())
        return ObjectReport(
            index=index,
            object_type=obj.type,
            label=obj.label,
            ok=points > 0,
            points=points,
            finite_fraction=finite_frac,
            in_bounds_fraction=1.0,
            message="" if points > 0 else "Surface produced no finite values",
        )
    except Exception as exc:
        return ObjectReport(
            index=index, object_type=obj.type, label=obj.label,
            ok=False, message=f"Render error: {exc}",
        )
