"""Export a rendered Figure to PNG / SVG / PDF bytes, and Scene to JSON."""
from __future__ import annotations

import io
import json

from matplotlib.figure import Figure

from schema.scene import Scene


def figure_to_bytes(fig: Figure, fmt: str, transparent: bool = False) -> bytes:
    """Serialize a matplotlib Figure to bytes in png/svg/pdf format."""
    if fmt not in ("png", "svg", "pdf"):
        raise ValueError(f"Unsupported export format: {fmt}")
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format=fmt,
        transparent=transparent,
        bbox_inches="tight",
        pad_inches=0.1,
        facecolor=fig.get_facecolor() if not transparent else "none",
    )
    return buf.getvalue()


def scene_to_json(scene: Scene) -> str:
    """Serialize a Scene to reproducible, pretty-printed JSON."""
    return json.dumps(scene.model_dump(mode="json"), indent=2)


def scene_from_json(text: str) -> Scene:
    """Parse Scene JSON (raises ValueError on bad JSON or schema)."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    return Scene.model_validate(data)
