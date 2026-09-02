"""Math-to-Image: natural language -> mathematical equations -> image.

Run with:  python -m streamlit run app.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import streamlit as st

from config.settings import get_settings
from export.exporter import figure_to_bytes, scene_from_json, scene_to_json
from llm.pipeline import generate, make_provider, render_validated
from math_engine.parser import to_latex
from schema.scene import (
    CartesianObject,
    ImplicitObject,
    ParametricObject,
    PolarObject,
    RegionObject,
    Scene,
    Surface3DObject,
)

st.set_page_config(page_title="Math-to-Image", page_icon=":triangular_ruler:", layout="wide")

st.markdown(
    """
    <style>
      .block-container { max-width: 1200px; padding-top: 2.5rem; }
      h1 { letter-spacing: -0.02em; }
      .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Equation display helpers
# --------------------------------------------------------------------------- #
def object_equations_latex(obj) -> list[str]:
    """Return LaTeX strings describing one scene object."""
    if isinstance(obj, CartesianObject):
        return [f"y = {to_latex(obj.equation, ['x'])}"]
    if isinstance(obj, ParametricObject):
        p = obj.parameter
        return [
            f"x({p}) = {to_latex(obj.x, [p])}",
            f"y({p}) = {to_latex(obj.y, [p])}",
        ]
    if isinstance(obj, PolarObject):
        return [f"r(\\theta) = {to_latex(obj.equation, ['theta'])}"]
    if isinstance(obj, ImplicitObject):
        return [f"{to_latex(obj.equation, ['x', 'y'])} = 0"]
    if isinstance(obj, RegionObject):
        op = {"<": "<", "<=": "\\le", ">": ">", ">=": "\\ge"}[obj.operator]
        return [f"{to_latex(obj.equation, ['x', 'y'])} {op} 0"]
    if isinstance(obj, Surface3DObject):
        return [f"z = {to_latex(obj.equation, ['x', 'y'])}"]
    if obj.type == "point":
        return [f"({obj.x:g},\\; {obj.y:g})"]
    if obj.type == "segment":
        return [
            f"({obj.start[0]:g},\\; {obj.start[1]:g}) \\to ({obj.end[0]:g},\\; {obj.end[1]:g})"
        ]
    if obj.type == "text":
        return [f"\\text{{{obj.text}}} \\;\\; \\text{{at}} \\; ({obj.x:g},\\, {obj.y:g})"]
    return []


TYPE_NAMES = {
    "cartesian": "Cartesian function",
    "parametric": "Parametric curve",
    "polar": "Polar curve",
    "implicit": "Implicit curve",
    "region": "Inequality region",
    "surface3d": "3D surface",
    "point": "Point",
    "segment": "Line segment",
    "text": "Label",
}


def show_result(scene: Scene, fig, report) -> None:
    """Render image, equations, stats and downloads for a finished scene."""
    left, right = st.columns([7, 5], gap="large")

    with left:
        st.subheader("Generated Image")
        transparent = scene.render.background.lower() == "transparent"
        png = figure_to_bytes(fig, "png", transparent=transparent)
        st.image(png, use_container_width=True)

        st.caption("Every pixel above is computed from the equations on the right.")

        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("PNG", png, file_name="math-image.png", mime="image/png",
                           use_container_width=True)
        d2.download_button("SVG", figure_to_bytes(fig, "svg", transparent=transparent),
                           file_name="math-image.svg", mime="image/svg+xml",
                           use_container_width=True)
        d3.download_button("PDF", figure_to_bytes(fig, "pdf", transparent=transparent),
                           file_name="math-image.pdf", mime="application/pdf",
                           use_container_width=True)
        d4.download_button("JSON", scene_to_json(scene), file_name="scene.json",
                           mime="application/json", use_container_width=True)

    with right:
        st.subheader("Mathematical Representation")
        st.markdown(f"**{scene.title}**")
        for i, obj in enumerate(scene.objects):
            name = TYPE_NAMES.get(obj.type, obj.type)
            label = f" — {obj.label}" if obj.label else ""
            st.markdown(f"**Object {i + 1}** · {name}{label}")
            for eq in object_equations_latex(obj):
                st.latex(eq)

        st.subheader("Rendering Information")
        stats = st.columns(2)
        stats[0].metric("Coordinate system", scene.coordinate_system.capitalize())
        stats[1].metric("Status", "Success" if report.success else "Failed")
        stats2 = st.columns(4)
        stats2[0].metric("Objects", report.n_objects)
        stats2[1].metric("Valid", report.n_valid)
        stats2[2].metric("Invalid", report.n_invalid)
        stats2[3].metric("Finite samples", f"{report.finite_percent:.1f}%")
        st.caption(
            f"Resolution: {scene.render.width} × {scene.render.height} px · "
            f"Bounds: [{scene.bounds[0]:g}, {scene.bounds[1]:g}] × "
            f"[{scene.bounds[2]:g}, {scene.bounds[3]:g}]"
        )
        for w in report.warnings:
            st.warning(w)

        with st.expander("Scene JSON"):
            st.code(scene_to_json(scene), language="json")


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
st.title("Math-to-Image")
st.markdown(
    "Describe an image; a language model translates it into **pure mathematics** "
    "(equations, curves, regions), which a deterministic engine validates and renders. "
    "No image-generation model is ever used: *text → mathematics → geometry → pixels*."
)

settings = get_settings()
provider = make_provider(settings)

tab_generate, tab_json = st.tabs(["Generate from prompt", "Render from JSON"])

# ---- Tab 1: prompt -> LLM -> scene -> image ------------------------------- #
with tab_generate:
    if not provider.is_configured():
        st.error(
            f"LLM provider '{provider.name}' is not configured. "
            "Set the required API key in your environment (see .env.example). "
            "You can still render saved scenes in the 'Render from JSON' tab."
        )

    with st.form("generate_form"):
        prompt = st.text_area(
            "Describe the image you want",
            placeholder="Create a mathematical drawing of a butterfly...",
            height=90,
            label_visibility="collapsed",
        )
        c1, c2 = st.columns([1, 5])
        submitted = c1.form_submit_button("Generate", type="primary", use_container_width=True)
        c2.caption(
            f"Provider: {provider.name} · Repair attempts: {settings.max_repair_attempts} · "
            "The LLM outputs only a strict JSON scene; equations are validated with SymPy "
            "before anything is drawn."
        )

    examples = [
        "Create a mathematical drawing of a heart.",
        "Draw a circle tangent to a line, with the tangent point marked.",
        "Draw a butterfly curve.",
        "Create a mathematical drawing of a mountain landscape with a sun.",
        "Draw a 3D Gaussian surface.",
    ]
    st.caption("Try: " + " · ".join(f"*{e}*" for e in examples))

    if submitted:
        if not prompt.strip():
            st.warning("Enter a description first.")
        elif not provider.is_configured():
            st.error("Cannot generate: the LLM provider is not configured.")
        else:
            with st.status("Generating mathematical scene...", expanded=True) as status:
                st.write("Asking the model for a mathematical representation...")
                result = generate(prompt.strip(), provider=provider)
                for log in result.attempts:
                    if log.stage == "success":
                        st.write(f"Attempt {log.attempt}: scene validated and rendered.")
                    else:
                        st.write(f"Attempt {log.attempt} failed at {log.stage} stage:")
                        st.code(log.detail[:2000])
                        if log.attempt < settings.max_repair_attempts:
                            st.write("Attempting automatic repair...")
                if result.success:
                    status.update(label="Scene generated", state="complete", expanded=False)
                else:
                    status.update(label="Generation failed", state="error", expanded=True)

            if result.success and result.scene and result.figure:
                st.session_state["last_result"] = result
            elif not result.success:
                st.error(
                    "Unable to generate a valid mathematical representation after "
                    f"{settings.max_repair_attempts} attempts."
                )
                if result.error:
                    with st.expander("Last error"):
                        st.code(result.error)

    last = st.session_state.get("last_result")
    if last is not None and last.success:
        st.divider()
        show_result(last.scene, last.figure, last.render_report)

# ---- Tab 2: scene.json -> image (no LLM) ----------------------------------- #
with tab_json:
    st.markdown(
        "Rendering is fully **reproducible without the LLM**: paste or upload a "
        "`scene.json` exported earlier and the deterministic engine re-creates the image."
    )
    uploaded = st.file_uploader("Upload scene.json", type=["json"])
    default_text = uploaded.read().decode("utf-8") if uploaded else ""
    json_text = st.text_area(
        "Scene JSON", value=default_text, height=260,
        placeholder='{"title": "Circle", "bounds": [-5, 5, -5, 5], "objects": [{"type": "implicit", "equation": "x^2 + y^2 - 4"}]}',
    )
    if st.button("Render scene", type="primary"):
        if not json_text.strip():
            st.warning("Paste scene JSON or upload a file first.")
        else:
            try:
                scene = scene_from_json(json_text)
            except ValueError as exc:
                st.error(f"Invalid scene: {exc}")
            else:
                fig, report, error_text = render_validated(scene)
                if error_text is not None:
                    st.error("Mathematical validation or rendering failed:")
                    st.code(error_text)
                else:
                    show_result(scene, fig, report)

plt.close("all")
