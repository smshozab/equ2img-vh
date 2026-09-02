"""End-to-end tests over the bundled example scenes and serialization."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from export.exporter import figure_to_bytes, scene_from_json, scene_to_json
from llm.pipeline import render_validated
from schema.scene import Scene

EXAMPLES = sorted((Path(__file__).parent.parent / "examples").glob("*.json"))


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_example_scene_renders(path: Path):
    scene = scene_from_json(path.read_text())
    fig, report, error = render_validated(scene)
    assert error is None, f"{path.name}: {error}"
    assert report.success
    png = figure_to_bytes(fig, "png")
    assert len(png) > 1000


def test_scene_roundtrip_serialization():
    scene = Scene.model_validate({
        "title": "Roundtrip",
        "bounds": [-3, 3, -3, 3],
        "objects": [
            {"type": "parametric", "x": "cos(t)", "y": "sin(t)", "parameter": "t"},
            {"type": "text", "text": "unit circle", "x": 0, "y": 0},
        ],
    })
    text = scene_to_json(scene)
    restored = scene_from_json(text)
    assert restored == scene
    # And valid JSON with stable keys
    data = json.loads(text)
    assert data["title"] == "Roundtrip"
    assert data["objects"][0]["type"] == "parametric"


def test_reproducible_without_llm():
    """scene.json -> image must never require an LLM."""
    text = (Path(__file__).parent.parent / "examples" / "heart.json").read_text()
    scene = scene_from_json(text)
    fig1, r1, e1 = render_validated(scene)
    fig2, r2, e2 = render_validated(scene)
    assert e1 is None and e2 is None
    png1 = figure_to_bytes(fig1, "png")
    png2 = figure_to_bytes(fig2, "png")
    assert png1 == png2  # deterministic rendering


def test_invalid_json_rejected():
    with pytest.raises(ValueError, match="Invalid JSON"):
        scene_from_json("{not json")


def test_invalid_schema_rejected():
    with pytest.raises(Exception):
        scene_from_json('{"objects": [{"type": "nope"}]}')
