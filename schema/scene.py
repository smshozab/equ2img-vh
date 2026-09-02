"""Mathematical Scene DSL - Pydantic models.

The LLM produces JSON matching these models. Rendering is fully
deterministic given a validated Scene, no LLM required.
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

Number = float


class StyleModel(BaseModel):
    """Visual style for a scene object."""

    color: str = Field(default="#1a1a2e", description="Hex color or matplotlib color name")
    line_width: float = Field(default=2.0, ge=0.1, le=20)
    line_style: Literal["solid", "dashed", "dotted", "dashdot"] = "solid"
    alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    fill: bool = False
    fill_color: str | None = None
    z_order: int = 2

    @property
    def mpl_linestyle(self) -> str:
        return {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}[self.line_style]


class BaseObject(BaseModel):
    label: str | None = Field(default=None, description="Optional human-readable name")
    style: StyleModel = Field(default_factory=StyleModel)


class CartesianObject(BaseObject):
    """y = f(x)"""

    type: Literal["cartesian"] = "cartesian"
    equation: str = Field(description="Expression in x, e.g. 'sin(x)'")
    domain: list[Number] = Field(default=[-10.0, 10.0], min_length=2, max_length=2)
    samples: int = Field(default=2000, ge=16, le=20000)

    @field_validator("domain")
    @classmethod
    def _domain_ordered(cls, v: list[Number]) -> list[Number]:
        if v[0] >= v[1]:
            raise ValueError(f"domain must be [min, max] with min < max, got {v}")
        return v


class ParametricObject(BaseObject):
    """x = f(t), y = g(t)"""

    type: Literal["parametric"] = "parametric"
    x: str = Field(description="Expression in the parameter, e.g. 'cos(t)'")
    y: str = Field(description="Expression in the parameter, e.g. 'sin(t)'")
    parameter: str = Field(default="t")
    range: list[Number] = Field(default=[0.0, 6.283185307179586], min_length=2, max_length=2)
    samples: int = Field(default=4000, ge=16, le=50000)

    @field_validator("range")
    @classmethod
    def _range_ordered(cls, v: list[Number]) -> list[Number]:
        if v[0] >= v[1]:
            raise ValueError(f"range must be [min, max] with min < max, got {v}")
        return v


class PolarObject(BaseObject):
    """r = f(theta)"""

    type: Literal["polar"] = "polar"
    equation: str = Field(description="Expression in theta, e.g. '1 + 0.5*cos(5*theta)'")
    range: list[Number] = Field(default=[0.0, 6.283185307179586], min_length=2, max_length=2)
    samples: int = Field(default=4000, ge=16, le=50000)
    center: list[Number] = Field(default=[0.0, 0.0], min_length=2, max_length=2)

    @field_validator("range")
    @classmethod
    def _range_ordered(cls, v: list[Number]) -> list[Number]:
        if v[0] >= v[1]:
            raise ValueError(f"range must be [min, max] with min < max, got {v}")
        return v


class ImplicitObject(BaseObject):
    """F(x, y) = 0, rendered via zero-contour extraction."""

    type: Literal["implicit"] = "implicit"
    equation: str = Field(description="Expression in x and y; the curve is where it equals 0")
    domain: DomainXY | None = None
    resolution: int = Field(default=600, ge=32, le=1500)


class RegionObject(BaseObject):
    """Inequality region F(x, y) < 0 (or > 0)."""

    type: Literal["region"] = "region"
    equation: str = Field(description="Expression in x and y")
    operator: Literal["<", "<=", ">", ">="] = "<"
    domain: DomainXY | None = None
    resolution: int = Field(default=400, ge=32, le=1000)


class Surface3DObject(BaseObject):
    """z = f(x, y)"""

    type: Literal["surface3d"] = "surface3d"
    equation: str = Field(description="Expression in x and y giving z")
    domain: DomainXY | None = None
    resolution: int = Field(default=120, ge=8, le=400)
    mode: Literal["surface", "wireframe", "contour"] = "surface"
    colormap: str = "viridis"


class PointObject(BaseObject):
    type: Literal["point"] = "point"
    x: Number
    y: Number
    marker: str = Field(default="o")
    size: float = Field(default=60.0, ge=1, le=1000)


class SegmentObject(BaseObject):
    type: Literal["segment"] = "segment"
    start: list[Number] = Field(min_length=2, max_length=2)
    end: list[Number] = Field(min_length=2, max_length=2)


class TextObject(BaseObject):
    type: Literal["text"] = "text"
    text: str = Field(max_length=200)
    x: Number
    y: Number
    font_size: float = Field(default=12.0, ge=4, le=72)


class DomainXY(BaseModel):
    x: list[Number] = Field(default=[-5.0, 5.0], min_length=2, max_length=2)
    y: list[Number] = Field(default=[-5.0, 5.0], min_length=2, max_length=2)

    @model_validator(mode="after")
    def _ordered(self) -> "DomainXY":
        if self.x[0] >= self.x[1]:
            raise ValueError(f"domain.x must be [min, max] with min < max, got {self.x}")
        if self.y[0] >= self.y[1]:
            raise ValueError(f"domain.y must be [min, max] with min < max, got {self.y}")
        return self


SceneObject = Annotated[
    Union[
        CartesianObject,
        ParametricObject,
        PolarObject,
        ImplicitObject,
        RegionObject,
        Surface3DObject,
        PointObject,
        SegmentObject,
        TextObject,
    ],
    Field(discriminator="type"),
]


class RenderSettings(BaseModel):
    width: int = Field(default=2000, ge=100, le=6000, description="Pixels")
    height: int = Field(default=2000, ge=100, le=6000, description="Pixels")
    dpi: int = Field(default=200, ge=50, le=600)
    background: str = Field(default="#ffffff", description="'transparent' or a color")
    show_axes: bool = False
    show_grid: bool = False
    equal_aspect: bool = True


class Scene(BaseModel):
    """Top-level Mathematical Scene."""

    title: str = Field(default="Untitled", max_length=200)
    coordinate_system: Literal["cartesian", "polar", "3d"] = "cartesian"
    bounds: list[Number] = Field(
        default=[-10.0, 10.0, -10.0, 10.0],
        min_length=4,
        max_length=4,
        description="[xmin, xmax, ymin, ymax]",
    )
    objects: list[SceneObject] = Field(min_length=1, max_length=64)
    render: RenderSettings = Field(default_factory=RenderSettings)

    @field_validator("bounds")
    @classmethod
    def _bounds_ordered(cls, v: list[Number]) -> list[Number]:
        if v[0] >= v[1] or v[2] >= v[3]:
            raise ValueError(f"bounds must be [xmin, xmax, ymin, ymax] with min < max, got {v}")
        return v

    @property
    def is_3d(self) -> bool:
        return self.coordinate_system == "3d" or any(
            getattr(o, "type", "") == "surface3d" for o in self.objects
        )


# Resolve forward references
ImplicitObject.model_rebuild()
RegionObject.model_rebuild()
Surface3DObject.model_rebuild()
Scene.model_rebuild()
