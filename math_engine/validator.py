"""Scene-level validation: schema -> expressions -> numerical safety."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydantic import ValidationError

from math_engine.parser import ExpressionError, parse_expression
from schema.scene import (
    CartesianObject,
    ImplicitObject,
    ParametricObject,
    PolarObject,
    RegionObject,
    Scene,
    Surface3DObject,
)

MAX_ABS_VALUE = 1e9  # values beyond this are treated as numerically unsafe


@dataclass
class ObjectIssue:
    index: int
    object_type: str
    label: str | None
    message: str

    def __str__(self) -> str:
        name = f" ('{self.label}')" if self.label else ""
        return f"Object {self.index + 1} [{self.object_type}]{name}: {self.message}"


@dataclass
class ValidationReport:
    valid: bool
    issues: list[ObjectIssue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error_text(self) -> str:
        return "\n".join(str(i) for i in self.issues)


def validate_scene_dict(data: dict) -> tuple[Scene | None, str | None]:
    """Pydantic schema validation. Returns (scene, error_message)."""
    try:
        return Scene.model_validate(data), None
    except ValidationError as exc:
        lines = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"])
            lines.append(f"{loc}: {err['msg']}")
        return None, "Schema validation failed:\n" + "\n".join(lines)


def _check_finite_fraction(values: np.ndarray, what: str, issues_out: list[str]) -> None:
    finite = np.isfinite(values)
    frac = float(finite.mean()) if values.size else 0.0
    if frac == 0.0:
        issues_out.append(f"{what} produces no finite values over its domain")
    if finite.any():
        biggest = float(np.nanmax(np.abs(np.where(finite, values, 0.0))))
        if biggest > MAX_ABS_VALUE:
            issues_out.append(
                f"{what} produces extremely large values (max |v| ~ {biggest:.2e})"
            )


def validate_scene(scene: Scene) -> ValidationReport:
    """Expression + numerical validation for every object in the scene."""
    report = ValidationReport(valid=True)

    for i, obj in enumerate(scene.objects):
        problems: list[str] = []
        try:
            if isinstance(obj, CartesianObject):
                parsed = parse_expression(obj.equation, ["x"])
                xs = np.linspace(obj.domain[0], obj.domain[1], 512)
                _check_finite_fraction(parsed.evaluate(xs), "y = f(x)", problems)

            elif isinstance(obj, ParametricObject):
                p = obj.parameter
                if not p.isidentifier() or len(p) > 8:
                    problems.append(f"Invalid parameter name '{p}'")
                else:
                    px = parse_expression(obj.x, [p])
                    py = parse_expression(obj.y, [p])
                    ts = np.linspace(obj.range[0], obj.range[1], 512)
                    _check_finite_fraction(px.evaluate(ts), "x(t)", problems)
                    _check_finite_fraction(py.evaluate(ts), "y(t)", problems)

            elif isinstance(obj, PolarObject):
                parsed = parse_expression(obj.equation, ["theta"])
                ths = np.linspace(obj.range[0], obj.range[1], 512)
                _check_finite_fraction(parsed.evaluate(ths), "r(theta)", problems)

            elif isinstance(obj, (ImplicitObject, RegionObject)):
                parsed = parse_expression(obj.equation, ["x", "y"])
                dom = obj.domain
                x0, x1 = (dom.x if dom else (scene.bounds[0], scene.bounds[1]))
                y0, y1 = (dom.y if dom else (scene.bounds[2], scene.bounds[3]))
                xs = np.linspace(x0, x1, 64)
                ys = np.linspace(y0, y1, 64)
                X, Y = np.meshgrid(xs, ys)
                _check_finite_fraction(parsed.evaluate(X, Y), "F(x, y)", problems)

            elif isinstance(obj, Surface3DObject):
                parsed = parse_expression(obj.equation, ["x", "y"])
                dom = obj.domain
                x0, x1 = (dom.x if dom else (scene.bounds[0], scene.bounds[1]))
                y0, y1 = (dom.y if dom else (scene.bounds[2], scene.bounds[3]))
                xs = np.linspace(x0, x1, 32)
                ys = np.linspace(y0, y1, 32)
                X, Y = np.meshgrid(xs, ys)
                _check_finite_fraction(parsed.evaluate(X, Y), "z = f(x, y)", problems)

            # point / segment / text: pure numbers already validated by Pydantic
            else:
                for attr in ("x", "y"):
                    v = getattr(obj, attr, None)
                    if isinstance(v, float) and not np.isfinite(v):
                        problems.append(f"Coordinate {attr} is not finite")

        except ExpressionError as exc:
            problems.append(str(exc))
        except Exception as exc:  # defensive: never crash validation
            problems.append(f"Unexpected validation error: {exc}")

        for msg in problems:
            report.issues.append(
                ObjectIssue(index=i, object_type=obj.type, label=obj.label, message=msg)
            )

    # Mixed 2D/3D check
    has_3d = any(o.type == "surface3d" for o in scene.objects)
    has_2d = any(o.type not in ("surface3d",) for o in scene.objects)
    if has_3d and has_2d:
        report.warnings.append(
            "Scene mixes 3D surfaces with 2D objects; 2D objects will be skipped in 3D mode"
        )

    report.valid = len(report.issues) == 0
    return report
