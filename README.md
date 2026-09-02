# Math-to-Image

Convert natural-language descriptions of images into **pure mathematics** — equations,
curves, regions, surfaces — and render them into images.

```
Text → Mathematics → Geometry → Pixels
```

No image-generation model is ever used. An LLM produces only a strict JSON
"Mathematical Scene"; a deterministic Python engine validates the mathematics and
renders it with NumPy + Matplotlib. The equations are the source of truth.

## Example

Prompt: *"Create a mathematical drawing of a heart."*

Generated mathematics:

```
x(t) = 16 sin(t)^3
y(t) = 13 cos(t) - 5 cos(2t) - 2 cos(3t) - cos(4t)
```

The engine samples the curve, splits discontinuities, renders it, and reports quality
statistics — and shows the equations next to the image.

## Architecture

```
User Prompt
  → LLM (interprets the visual concept)
  → Mathematical Scene JSON (strict DSL, no code)
  → Schema validation           (Pydantic)
  → Expression validation       (SymPy safe parsing, allowlist)
  → Numerical safety checks     (NaN / Inf / magnitude)
  → Sampling & rendering        (NumPy + Matplotlib)
  → Image + equations + quality report
```

If validation or rendering fails, the exact error is sent back to the LLM in a
**repair loop** (max 3 attempts).

```
math-to-image/
├── app.py                  # Streamlit UI
├── config/settings.py      # env-based configuration
├── llm/
│   ├── base.py             # provider interface + JSON extraction
│   ├── gateway.py          # Vercel AI Gateway (default)
│   ├── gemini.py           # Google Gemini
│   ├── openai.py           # OpenAI
│   ├── prompts.py          # system + repair prompts
│   └── pipeline.py         # generate -> validate -> render -> repair loop
├── schema/scene.py         # Mathematical Scene DSL (Pydantic)
├── math_engine/
│   ├── safety.py           # allowlist + forbidden-token prechecks
│   ├── parser.py           # SymPy parsing -> NumPy lambdify (no eval)
│   ├── validator.py        # per-object expression + numeric validation
│   └── sampler.py          # discontinuity-aware sampling, grid sampling
├── renderer/               # cartesian, parametric, polar, implicit, surface3d
├── export/exporter.py      # PNG / SVG / PDF / scene JSON
├── examples/               # deterministic example scenes
└── tests/                  # pytest suite (no LLM required)
```

## Installation

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your API key
python -m streamlit run app.py
```

## Environment variables

| Variable | Description |
| --- | --- |
| `LLM_PROVIDER` | `gateway` (default), `gemini`, or `openai` |
| `AI_GATEWAY_API_KEY` | Vercel AI Gateway key (default provider) |
| `AI_GATEWAY_MODEL` | Gateway model id, e.g. `openai/gpt-4o-mini` |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | Google Gemini configuration |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | OpenAI configuration |
| `MAX_REPAIR_ATTEMPTS` | Repair loop retries (default 3) |

## Mathematical Scene DSL

A scene is a JSON document:

```json
{
  "title": "Circle",
  "coordinate_system": "cartesian",
  "bounds": [-5, 5, -5, 5],
  "objects": [
    { "type": "implicit", "equation": "x^2 + y^2 - 4",
      "domain": { "x": [-5, 5], "y": [-5, 5] } }
  ]
}
```

Supported object types:

| Type | Meaning | Key fields |
| --- | --- | --- |
| `cartesian` | y = f(x) | `equation`, `domain`, `samples` |
| `parametric` | x = f(t), y = g(t) | `x`, `y`, `parameter`, `range` |
| `polar` | r = f(θ) | `equation`, `range`, `center` |
| `implicit` | F(x, y) = 0 (zero contour) | `equation`, `domain`, `resolution` |
| `region` | F(x, y) < 0 shaded | `equation`, `operator` |
| `surface3d` | z = f(x, y) | `equation`, `mode`, `colormap` |
| `point` | (x, y) | `x`, `y`, `marker`, `size` |
| `segment` | (x1,y1) → (x2,y2) | `start`, `end` |
| `text` | label | `text`, `x`, `y` |

Each object accepts `label` and `style` (`color`, `line_width`, `line_style`,
`alpha`, `fill`, `fill_color`, `z_order`). Scene-level `render` controls size,
dpi, background (including `transparent`), axes, and grid.

## Safety model

LLM output is treated as untrusted data, never as code:

1. **No `eval()`** — expressions go through SymPy `parse_expr` with a restricted
   local dict only.
2. **Token precheck** — `__`, `import`, `lambda`, quotes, and other non-math
   characters are rejected before parsing.
3. **Function allowlist** — only `sin cos tan asin acos atan atan2 sqrt exp log
   abs sinh cosh tanh floor ceil sign min max` plus constants `pi`, `e`, `tau`.
4. **Variable allowlist per context** — `x` for cartesian, `x, y` for
   implicit/region/surface, the declared parameter for parametric, `theta` for polar.
   Undefined symbols are validation errors.
5. **Complexity limits** — max expression length and AST size.
6. **Numeric checks** — NaN / Infinity / huge-magnitude detection before rendering;
   discontinuities split rather than drawn across.

## Reproducibility

Generation and rendering are strictly separated:

- `prompt → scene.json → image` (uses the LLM once)
- `scene.json → image` (never uses an LLM — see the "Render from JSON" tab)

Download the scene JSON from the UI and re-render it any time; rendering is
byte-for-byte deterministic.

## Adding a renderer

1. Add a Pydantic model in `schema/scene.py` and register it in `SceneObject`.
2. Add validation in `math_engine/validator.py`.
3. Add a `renderer/<type>.py` returning `(segments, finite_fraction)` and wire it
   into `renderer/base.py`.
4. Describe the type in `llm/prompts.py` so the LLM can use it.
5. Add tests.

## Adding an LLM provider

Implement `LLMProvider.complete(system, user) -> str` in `llm/<name>.py` and
register it in `llm/pipeline.make_provider`. See `llm/gemini.py` for a minimal
example.

## Testing

```bash
python -m pytest
```

The suite covers the parser (including malicious input), validators, all
renderers, NaN/discontinuity handling, multi-object scenes, exports (PNG/SVG/PDF),
serialization round-trips, and the bundled example scenes — no API key required.

## Examples

`examples/` contains ready-to-render scenes: `circle.json`, `heart.json`,
`butterfly.json`, `spiral.json`, `waves.json`. Paste any of them into the
"Render from JSON" tab.

## Roadmap

- Advanced mode: Fourier series, Bézier curves, splines, L-systems, boolean
  regions, transformations
- Recognizability scoring and equation-complexity metrics
- A benchmark harness for evaluating how well LLMs convert visual concepts into
  executable mathematics
