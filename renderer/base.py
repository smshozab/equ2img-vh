"""Renderer orchestration: Scene -> matplotlib Figure + quality report.

Rendering is fully deterministic and never calls an LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from renderer.cartesian import render_cartesian
from renderer.implicit import render_implicit, render_region
from renderer.parametric import render_parametric
from renderer.polar import render_polar
from renderer.surface3d import render_surface3d
from schema.scene import (
    CartesianObject,
    ImplicitObject,
    ParametricObject,
    PointObject,
    PolarObject,
    RegionObject,
    Scene,
    SegmentObject,
    Surface3DObject,
    TextObject,
)


@dataclass
class ObjectReport:
    index: int
    object_type: str
    label: str | None
    ok: bool
    points: int = 0
    finite_fraction: float = 1.0
    in_bounds_fraction: float = 1.0
    message: str = ""


@dataclass
class RenderReport:
    success: bool
    objects: list[ObjectReport] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_objects(self) -> int:
        return len(self.objects)

    @property
    def n_valid(self) -> int:
        return sum(1 for o in self.objects if o.ok)

    @property
    def n_invalid(self) -> int:
        return self.n_objects - self.n_valid

    @property
    def finite_percent(self) -> float:
        drawn = [o for o in self.objects if o.ok and o.points > 0]
        if not drawn:
            return 0.0
        return 100.0 * float(np.mean([o.finite_fraction for o in drawn]))

    @property
    def is_empty(self) -> bool:
        return all(o.points == 0 for o in self.objects)

    def error_text(self) -> str:
        lines = []
        for o in self.objects:
            if not o.ok:
                name = f" ('{o.label}')" if o.label else ""
                lines.append(f"Object {o.index + 1} [{o.object_type}]{name}: {o.message}")
        return "\n".join(lines)


def _in_bounds_fraction(segments: list[np.ndarray], bounds: list[float]) -> float:
    pts = np.vstack(segments) if segments else np.empty((0, 2))
    if pts.size == 0:
        return 0.0
    x0, x1, y0, y1 = bounds
    inside = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
    return float(inside.mean())


def render_scene(scene: Scene) -> tuple[Figure, RenderReport]:
    """Render a validated Scene to a matplotlib Figure with a quality report."""
    settings = scene.render
    figsize = (settings.width / settings.dpi, settings.height / settings.dpi)
    transparent = settings.background.lower() == "transparent"
    facecolor = "none" if transparent else settings.background

    fig = plt.figure(figsize=figsize, dpi=settings.dpi)
    fig.patch.set_facecolor(facecolor)
    if transparent:
        fig.patch.set_alpha(0.0)

    report = RenderReport(success=True)

    if scene.is_3d:
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(facecolor if not transparent else "none")
        for i, obj in enumerate(scene.objects):
            if isinstance(obj, Surface3DObject):
                report.objects.append(render_surface3d(ax, obj, scene, i))
            else:
                report.objects.append(
                    ObjectReport(
                        index=i,
                        object_type=obj.type,
                        label=obj.label,
                        ok=True,
                        points=0,
                        message="2D object skipped in 3D scene",
                    )
                )
                report.warnings.append(f"Object {i + 1} ({obj.type}) skipped in 3D scene")
        if not settings.show_axes:
            ax.set_axis_off()
    else:
        ax = fig.add_subplot(111)
        ax.set_facecolor(facecolor if not transparent else "none")
        for i, obj in enumerate(scene.objects):
            rep = _render_2d_object(ax, obj, scene, i)
            rep.in_bounds_fraction = rep.in_bounds_fraction  # computed inside
            report.objects.append(rep)

        ax.set_xlim(scene.bounds[0], scene.bounds[1])
        ax.set_ylim(scene.bounds[2], scene.bounds[3])
        if settings.equal_aspect:
            ax.set_aspect("equal", adjustable="box")
        if settings.show_grid:
            ax.grid(True, alpha=0.3, linewidth=0.5)
        if settings.show_axes:
            ax.axhline(0, color="#888888", linewidth=0.8, zorder=1)
            ax.axvline(0, color="#888888", linewidth=0.8, zorder=1)
        else:
            ax.set_axis_off()

    fig.tight_layout(pad=0.5)

    report.success = report.n_valid > 0 and not report.is_empty
    if report.is_empty:
        report.warnings.append("Rendered image is empty: no object produced any drawable points")
    return fig, report


def _render_2d_object(ax, obj, scene: Scene, index: int) -> ObjectReport:
    try:
        if isinstance(obj, CartesianObject):
            segments, finite = render_cartesian(ax, obj)
        elif isinstance(obj, ParametricObject):
            segments, finite = render_parametric(ax, obj)
        elif isinstance(obj, PolarObject):
            segments, finite = render_polar(ax, obj)
        elif isinstance(obj, ImplicitObject):
            segments, finite = render_implicit(ax, obj, scene)
        elif isinstance(obj, RegionObject):
            segments, finite = render_region(ax, obj, scene)
        elif isinstance(obj, PointObject):
            ax.scatter(
                [obj.x], [obj.y],
                s=obj.size, c=obj.style.color, marker=obj.marker,
                alpha=obj.style.alpha, zorder=obj.style.z_order + 1,
            )
            segments, finite = [np.array([[obj.x, obj.y], [obj.x, obj.y]])], 1.0
        elif isinstance(obj, SegmentObject):
            xs = [obj.start[0], obj.end[0]]
            ys = [obj.start[1], obj.end[1]]
            ax.plot(
                xs, ys,
                color=obj.style.color, linewidth=obj.style.line_width,
                linestyle=obj.style.mpl_linestyle, alpha=obj.style.alpha,
                zorder=obj.style.z_order,
            )
            segments, finite = [np.column_stack([xs, ys])], 1.0
        elif isinstance(obj, TextObject):
            ax.text(
                obj.x, obj.y, obj.text,
                fontsize=obj.font_size, color=obj.style.color,
                alpha=obj.style.alpha, ha="center", va="center",
                zorder=obj.style.z_order + 2,
            )
            segments, finite = [np.array([[obj.x, obj.y], [obj.x, obj.y]])], 1.0
        else:  # pragma: no cover
            return ObjectReport(
                index=index, object_type=obj.type, label=obj.label,
                ok=False, message=f"No renderer for object type '{obj.type}'",
            )

        points = int(sum(len(s) for s in segments))
        return ObjectReport(
            index=index,
            object_type=obj.type,
            label=obj.label,
            ok=points > 0,
            points=points,
            finite_fraction=finite,
            in_bounds_fraction=_in_bounds_fraction(segments, scene.bounds),
            message="" if points > 0 else "Object produced no drawable points",
        )
    except Exception as exc:
        return ObjectReport(
            index=index, object_type=obj.type, label=obj.label,
            ok=False, message=f"Render error: {exc}",
        )
