import ast
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from ultralytics.cfg import get_cfg
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils.callbacks.base import add_integration_callbacks, get_default_callbacks


def _reload_swanlab_callback(monkeypatch):
    fake_swanlab = SimpleNamespace(
        init=Mock(return_value=SimpleNamespace(id="new-run-id")),
        log=Mock(),
        finish=Mock(),
    )
    monkeypatch.setitem(sys.modules, "swanlab", fake_swanlab)
    sys.modules.pop("ultralytics.utils.callbacks.swanlab", None)
    return importlib.import_module("ultralytics.utils.callbacks.swanlab"), fake_swanlab


def test_swanlab_args_are_valid_yolo_cfg_keys():
    cfg = get_cfg(
        overrides={
            "swanlab": True,
            "swanlab_project": "YOLO",
            "swanlab_run_name": "exp1",
            "swanlab_mode": "local",
            "swanlab_log_interval": 5,
        }
    )

    assert cfg.swanlab is True
    assert cfg.swanlab_project == "YOLO"
    assert cfg.swanlab_run_name == "exp1"
    assert cfg.swanlab_mode == "local"
    assert cfg.swanlab_log_interval == 5


def test_train_script_exposes_swanlab_args():
    tree = ast.parse(Path("train.py").read_text(encoding="utf-8"))
    train_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "train"
    ]
    assert train_calls

    keyword_names = {keyword.arg for keyword in train_calls[0].keywords}
    assert {
        "swanlab",
        "swanlab_project",
        "swanlab_run_name",
        "swanlab_mode",
        "swanlab_log_interval",
    } <= keyword_names


def test_swanlab_callbacks_are_registered_for_trainers():
    class DummyTrainer:
        def __init__(self):
            self.callbacks = get_default_callbacks()

    trainer = DummyTrainer()

    add_integration_callbacks(trainer)

    assert any(
        callback.__module__ == "ultralytics.utils.callbacks.swanlab"
        for callback in trainer.callbacks["on_pretrain_routine_end"]
    )


def test_swanlab_init_resumes_existing_run_id(monkeypatch, tmp_path):
    callback, swanlab = _reload_swanlab_callback(monkeypatch)
    trainer = SimpleNamespace(
        args=SimpleNamespace(
            swanlab=True,
            swanlab_project="YOLO",
            swanlab_run_name="exp1",
            swanlab_mode="local",
        ),
        save_dir=tmp_path,
        swanlab_run_id="old-run-id",
    )

    callback.on_pretrain_routine_end(trainer)

    swanlab.init.assert_called_once()
    kwargs = swanlab.init.call_args.kwargs
    assert kwargs["project"] == "YOLO"
    assert kwargs["name"] == "exp1"
    assert kwargs["mode"] == "local"
    assert kwargs["id"] == "old-run-id"
    assert kwargs["resume"] == "must"
    assert kwargs["log_dir"] == str(tmp_path)
    assert trainer.swanlab_run_id == "new-run-id"
    assert trainer._swanlab is swanlab


def test_swanlab_batch_log_uses_loss_and_lr_groups(monkeypatch):
    callback, swanlab = _reload_swanlab_callback(monkeypatch)
    trainer = SimpleNamespace(
        args=SimpleNamespace(swanlab=True, swanlab_log_interval=10),
        _swanlab=swanlab,
        global_step=20,
        batch_i=3,
        epoch=1,
        loss=torch.tensor(6.0),
        loss_items=torch.tensor([1.0, 2.0, 3.0]),
        optimizer=SimpleNamespace(param_groups=[{"lr": 0.01}, {"lr": 0.001}]),
        label_loss_items=lambda loss_items, prefix="train": {
            "train/box_loss": loss_items[0],
            "train/cls_loss": loss_items[1],
            "train/dfl_loss": loss_items[2],
        },
    )

    callback.on_train_batch_end(trainer)

    swanlab.log.assert_called_once()
    payload = swanlab.log.call_args.args[0]
    assert swanlab.log.call_args.kwargs["step"] == 20
    assert payload["train/loss/total"] == 6.0
    assert payload["train/loss/box_loss"] == 1.0
    assert payload["train/loss/cls_loss"] == 2.0
    assert payload["train/lr/pg_0"] == 0.01
    assert payload["train/epoch"] == 1
    assert payload["train/step"] == 3


def test_swanlab_checkpoint_metadata_round_trips_run_id():
    trainer = SimpleNamespace(swanlab_run_id="run-123")

    assert BaseTrainer._swanlab_checkpoint_metadata(trainer) == {"swanlab_run_id": "run-123"}

    trainer.swanlab_run_id = None
    trainer.ema = None
    BaseTrainer._load_checkpoint_state(
        trainer,
        {
            "optimizer": None,
            "scaler": None,
            "ema": None,
            "best_fitness": 0.5,
            "swanlab_run_id": "run-456",
        },
    )

    assert trainer.best_fitness == 0.5
    assert trainer.swanlab_run_id == "run-456"
