# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.nn.tasks import parse_model


def _detect_legacy(model_cfg: dict) -> bool:
    model, _ = parse_model(model_cfg, ch=3, verbose=False)
    return model[-1].legacy


def _base_model_cfg(backbone: list) -> dict:
    return {
        "nc": 3,
        "depth_multiple": 1.0,
        "width_multiple": 1.0,
        "scale": "n",
        "backbone": backbone,
        "head": [[[0, 1, 2], 1, "Detect", ["nc"]]],
    }


def test_parse_model_uses_existing_legacy_inference_when_yaml_omits_legacy():
    cfg = _base_model_cfg(
        [
            [-1, 1, "Conv", [16, 3, 2]],
            [-1, 1, "Conv", [32, 3, 2]],
            [-1, 1, "C3k2", [64, False, 0.25]],
        ]
    )

    assert _detect_legacy(cfg) is False


def test_parse_model_allows_yaml_legacy_to_override_inference():
    cfg = _base_model_cfg(
        [
            [-1, 1, "Conv", [16, 3, 2]],
            [-1, 1, "Conv", [32, 3, 2]],
            [-1, 1, "C3k2", [64, False, 0.25]],
        ]
    )
    cfg["legacy"] = True

    assert _detect_legacy(cfg) is True


def test_parse_model_accepts_yaml_legacy_false_for_legacy_architectures():
    cfg = _base_model_cfg(
        [
            [-1, 1, "Conv", [16, 3, 2]],
            [-1, 1, "Conv", [32, 3, 2]],
            [-1, 1, "Conv", [64, 3, 2]],
        ]
    )
    cfg["legacy"] = False

    assert _detect_legacy(cfg) is False
