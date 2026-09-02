"""Tests for schema and scene validation."""
from math_engine.validator import validate_scene, validate_scene_dict
from schema.scene import Scene


def make_scene(objects: list[dict], **kwargs) -> Scene:
    data = {"title": "Test", "bounds": [-5, 5, -5, 5], "objects": objects}
    data.update(kwargs)
    return Scene.model_validate(data)


class TestSchemaValidation:
    def test_valid_minimal(self):
        scene, err = validate_scene_dict(
            {"objects": [{"type": "implicit", "equation": "x^2 + y^2 - 4"}]}
        )
        assert err is None
        assert scene is not None

    def test_unknown_type_rejected(self):
        scene, err = validate_scene_dict(
            {"objects": [{"type": "hologram", "equation": "x"}]}
        )
        assert scene is None
        assert "objects" in err

    def test_bad_bounds_rejected(self):
        scene, err = validate_scene_dict(
            {"bounds": [5, -5, -5, 5], "objects": [{"type": "cartesian", "equation": "x"}]}
        )
        assert scene is None

    def test_empty_objects_rejected(self):
        scene, err = validate_scene_dict({"objects": []})
        assert scene is None

    def test_bad_domain_rejected(self):
        scene, err = validate_scene_dict(
            {"objects": [{"type": "cartesian", "equation": "x", "domain": [3, 3]}]}
        )
        assert scene is None


class TestExpressionValidation:
    def test_valid_scene_passes(self):
        scene = make_scene([
            {"type": "cartesian", "equation": "sin(x)", "domain": [-5, 5]},
            {"type": "parametric", "x": "cos(t)", "y": "sin(t)", "parameter": "t"},
            {"type": "polar", "equation": "1 + 0.5*cos(5*theta)"},
            {"type": "implicit", "equation": "x^2 + y^2 - 4"},
        ])
        report = validate_scene(scene)
        assert report.valid, report.error_text()

    def test_undefined_variable_flagged(self):
        scene = make_scene([{"type": "cartesian", "equation": "sin(theta)"}])
        report = validate_scene(scene)
        assert not report.valid
        assert "theta" in report.error_text()
        assert "Object 1" in report.error_text()

    def test_parametric_wrong_parameter_flagged(self):
        scene = make_scene([
            {"type": "parametric", "x": "cos(u)", "y": "sin(u)", "parameter": "t"}
        ])
        report = validate_scene(scene)
        assert not report.valid

    def test_all_nan_flagged(self):
        # sqrt of strictly negative values everywhere on domain
        scene = make_scene([
            {"type": "cartesian", "equation": "sqrt(-1 - x^2)", "domain": [-2, 2]}
        ])
        report = validate_scene(scene)
        assert not report.valid
        assert "no finite values" in report.error_text()

    def test_huge_values_flagged(self):
        scene = make_scene([
            {"type": "cartesian", "equation": "exp(exp(x))", "domain": [4, 6]}
        ])
        report = validate_scene(scene)
        assert not report.valid

    def test_multiple_objects_issues_reported_per_object(self):
        scene = make_scene([
            {"type": "cartesian", "equation": "sin(x)"},
            {"type": "cartesian", "equation": "sin(w)"},
        ])
        report = validate_scene(scene)
        assert not report.valid
        assert "Object 2" in report.error_text()
        assert "Object 1" not in report.error_text()
