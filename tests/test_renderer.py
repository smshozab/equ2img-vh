"""Deterministic renderer tests (no LLM required)."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from export.exporter import figure_to_bytes
from renderer.base import render_scene
from schema.scene import Scene


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def render(objects: list[dict], **kwargs):
    data = {"title": "Test", "bounds": [-5, 5, -5, 5], "objects": objects}
    data.update(kwargs)
    scene = Scene.model_validate(data)
    return render_scene(scene)


class TestBasicShapes:
    def test_implicit_circle(self):
        fig, report = render([{"type": "implicit", "equation": "x^2 + y^2 - 4"}])
        assert report.success
        assert report.n_valid == 1
        assert report.objects[0].points > 50
        assert report.finite_percent > 99

    def test_sine_wave(self):
        fig, report = render(
            [{"type": "cartesian", "equation": "sin(x)", "domain": [-5, 5]}]
        )
        assert report.success
        assert report.objects[0].in_bounds_fraction > 0.99

    def test_parabola(self):
        fig, report = render(
            [{"type": "cartesian", "equation": "x^2", "domain": [-2, 2]}]
        )
        assert report.success

    def test_parametric_circle(self):
        fig, report = render([
            {"type": "parametric", "x": "2*cos(t)", "y": "2*sin(t)",
             "parameter": "t", "range": [0, 6.2832]}
        ])
        assert report.success
        # circle should stay fully in bounds
        assert report.objects[0].in_bounds_fraction == 1.0

    def test_polar_spiral(self):
        fig, report = render([
            {"type": "polar", "equation": "0.1*theta", "range": [0, 25]}
        ])
        assert report.success

    def test_region(self):
        fig, report = render([
            {"type": "region", "equation": "x^2 + y^2 - 4", "operator": "<"}
        ])
        assert report.success

    def test_point_segment_text(self):
        fig, report = render([
            {"type": "point", "x": 1, "y": 1},
            {"type": "segment", "start": [-3, -3], "end": [3, 3]},
            {"type": "text", "text": "Origin", "x": 0, "y": -0.5},
        ])
        assert report.success
        assert report.n_valid == 3

    def test_surface3d(self):
        fig, report = render(
            [{"type": "surface3d", "equation": "sin(sqrt(x^2 + y^2))", "resolution": 40}],
            coordinate_system="3d",
        )
        assert report.success


class TestRobustness:
    def test_discontinuity_split(self):
        # tan(x) has poles; rendering must succeed with split segments
        fig, report = render(
            [{"type": "cartesian", "equation": "tan(x)", "domain": [-4.5, 4.5]}]
        )
        assert report.success
        assert report.objects[0].finite_fraction < 1.01

    def test_partially_undefined(self):
        # log undefined for x <= 0: renders the defined part only
        fig, report = render(
            [{"type": "cartesian", "equation": "log(x)", "domain": [-2, 5]}]
        )
        assert report.success
        assert report.objects[0].finite_fraction < 0.9

    def test_multiple_objects(self):
        fig, report = render([
            {"type": "implicit", "equation": "x^2 + y^2 - 4"},
            {"type": "cartesian", "equation": "x", "domain": [-5, 5]},
            {"type": "point", "x": 1.414, "y": 1.414},
        ])
        assert report.success
        assert report.n_objects == 3
        assert report.n_valid == 3

    def test_empty_contour_reported(self):
        # x^2 + y^2 + 10 = 0 has no real solutions -> no curve
        fig, report = render([{"type": "implicit", "equation": "x^2 + y^2 + 10"}])
        assert not report.success
        assert report.is_empty


class TestExport:
    def test_png_export(self):
        fig, _ = render([{"type": "implicit", "equation": "x^2 + y^2 - 4"}])
        data = figure_to_bytes(fig, "png")
        assert data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_svg_export(self):
        fig, _ = render([{"type": "cartesian", "equation": "sin(x)"}])
        data = figure_to_bytes(fig, "svg")
        assert b"<svg" in data[:1000]

    def test_pdf_export(self):
        fig, _ = render([{"type": "cartesian", "equation": "sin(x)"}])
        data = figure_to_bytes(fig, "pdf")
        assert data[:5] == b"%PDF-"

    def test_bad_format_rejected(self):
        fig, _ = render([{"type": "cartesian", "equation": "sin(x)"}])
        with pytest.raises(ValueError):
            figure_to_bytes(fig, "exe")
