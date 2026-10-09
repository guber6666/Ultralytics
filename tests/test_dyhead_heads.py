from importlib import import_module
from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.tasks import parse_model


DYHEAD_HEADS = {
    "Detect_DyHead": {
        "args": (5, 16, 1, 16, False, (32, 64, 128)),
        "train_keys": {"boxes", "scores", "feats"},
    },
    "Segment_DyHead": {
        "args": (5, 8, 32, 16, 1, 16, False, (32, 64, 128)),
        "train_keys": {"boxes", "scores", "mask_coefficient", "proto", "feats"},
    },
    "Segment26_DyHead": {
        "args": (5, 8, 32, 16, 1, 1, True, (32, 64, 128)),
        "train_keys": {"one2many", "one2one"},
    },
    "OBB_DyHead": {
        "args": (5, 1, 16, 1, 16, False, (32, 64, 128)),
        "train_keys": {"boxes", "scores", "angle", "feats"},
    },
    "OBB26_DyHead": {
        "args": (5, 1, 16, 1, 1, True, (32, 64, 128)),
        "train_keys": {"one2many", "one2one"},
    },
    "Pose_DyHead": {
        "args": (5, (17, 3), 16, 1, 16, False, (32, 64, 128)),
        "train_keys": {"boxes", "scores", "kpts", "feats"},
    },
    "Pose26_DyHead": {
        "args": (5, (17, 3), 16, 1, 1, True, (32, 64, 128)),
        "train_keys": {"one2many", "one2one"},
    },
}

DYHEAD_PARSE_CASES = [
    ("Detect_DyHead", [5, 16, 1]),
    ("Segment_DyHead", [5, 32, 256, 16, 1]),
    ("Segment26_DyHead", [5, 32, 256, 16, 1]),
    ("OBB_DyHead", [5, 1, 16, 1]),
    ("OBB26_DyHead", [5, 1, 16, 1]),
    ("Pose_DyHead", [5, (17, 3), 16, 1]),
    ("Pose26_DyHead", [5, (17, 3), 16, 1]),
]

LQE_YAML_ROOT = Path("ultralytics/cfg/models/improve/head/LQE")
DYHEAD_YAML_ROOT = Path("ultralytics/cfg/models/improve/head/DyHead")

DYHEAD_YAMLS = [
    DYHEAD_YAML_ROOT / p.relative_to(LQE_YAML_ROOT).with_name(p.name.replace("LQE", "DyHead"))
    for p in sorted(LQE_YAML_ROOT.glob("*/*.yaml"))
]


def _feature_maps():
    return [
        torch.randn(2, 32, 32, 32),
        torch.randn(2, 64, 16, 16),
        torch.randn(2, 128, 8, 8),
    ]


def _set_stride(head):
    head.stride = torch.tensor([8.0, 16.0, 32.0])
    return head


@pytest.mark.parametrize("class_name", sorted(DYHEAD_HEADS))
def test_dyhead_module_exports_task_heads(class_name):
    module = import_module("ultralytics.nn.extra_modules.head.DyHead")
    assert hasattr(module, class_name)


@pytest.mark.parametrize(("module_name", "args"), DYHEAD_PARSE_CASES)
def test_parse_model_supports_dyhead_task_heads(module_name, args):
    model, save = parse_model(
        {
            "nc": 5,
            "kpt_shape": [17, 3],
            "depth_multiple": 1.0,
            "width_multiple": 1.0,
            "reg_max": 1 if "26" in module_name else 16,
            "end2end": "26" in module_name,
            "backbone": [
                [-1, 1, "Conv", [32, 3, 2]],
                [-1, 1, "Conv", [64, 3, 2]],
                [-1, 1, "Conv", [128, 3, 2]],
            ],
            "head": [
                [[0, 1, 2], 1, module_name, args],
            ],
        },
        ch=3,
        verbose=False,
    )

    assert len(model) == 4
    assert isinstance(save, list)


@pytest.mark.parametrize(("class_name", "config"), sorted(DYHEAD_HEADS.items()))
def test_dyhead_task_heads_train_outputs_include_task_specific_keys(class_name, config):
    module = import_module("ultralytics.nn.extra_modules.head.DyHead")
    head_cls = getattr(module, class_name)
    head = _set_stride(head_cls(*config["args"])).train()

    outputs = head(_feature_maps())

    assert isinstance(outputs, dict)
    assert config["train_keys"].issubset(outputs)
    if "one2many" in outputs:
        assert "boxes" in outputs["one2many"]
        assert "scores" in outputs["one2many"]
    if "one2one" in outputs:
        assert "boxes" in outputs["one2one"]
        assert "scores" in outputs["one2one"]


@pytest.mark.parametrize(("class_name", "config"), sorted(DYHEAD_HEADS.items()))
def test_dyhead_task_heads_eval_forward_runs(class_name, config):
    module = import_module("ultralytics.nn.extra_modules.head.DyHead")
    head_cls = getattr(module, class_name)
    head = _set_stride(head_cls(*config["args"])).eval()

    outputs = head(_feature_maps())

    assert outputs is not None


@pytest.mark.parametrize("yaml_path", DYHEAD_YAMLS)
def test_dyhead_yaml_examples_build(yaml_path):
    path = Path(yaml_path)
    assert path.is_file()
    model = YOLO(str(path))
    assert model.model is not None
