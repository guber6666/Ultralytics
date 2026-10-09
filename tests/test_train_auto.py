import sys

import train_auto


def test_build_commands_from_python_config_supports_common_train_args():
    cfg = train_auto.AutoTrainConfig(
        models=["ultralytics/cfg/models/11/yolo11n.yaml", "ultralytics/cfg/models/12/yolo12n.yaml"],
        data="/root/dataset/dataset_visdrone/data.yaml",
        epochs=300,
        batch=16,
        imgsz=640,
        device="0",
        project="train_auto",
        name_template="{stem}",
        swanlab=True,
        swanlab_project="YOLO",
        swanlab_run_name="experiment",
        swanlab_mode="local",
        swanlab_log_interval=5,
        extra={"optimizer": "SGD", "amp": False},
    )

    commands = train_auto.build_train_commands(cfg)

    assert commands[0][:4] == [sys.executable, "-m", "ultralytics.cfg", "train"]
    assert "model=ultralytics/cfg/models/11/yolo11n.yaml" in commands[0]
    assert "data=/root/dataset/dataset_visdrone/data.yaml" in commands[0]
    assert "epochs=300" in commands[0]
    assert "batch=16" in commands[0]
    assert "imgsz=640" in commands[0]
    assert "device=0" in commands[0]
    assert "project=train_auto" in commands[0]
    assert "name=yolo11n" in commands[0]
    assert "swanlab=True" in commands[0]
    assert "swanlab_project=YOLO" in commands[0]
    assert "swanlab_run_name=experiment" in commands[0]
    assert "swanlab_mode=local" in commands[0]
    assert "swanlab_log_interval=5" in commands[0]
    assert "optimizer=SGD" in commands[0]
    assert "amp=False" in commands[0]


def test_cli_args_override_editable_default_config():
    args = train_auto.parse_args(
        [
            "--models",
            "a.yaml",
            "b.yaml",
            "--data",
            "data.yaml",
            "--epochs",
            "2",
            "--batch",
            "4",
            "--imgsz",
            "320",
            "--device",
            "0",
            "--project",
            "runs_auto",
            "--name-template",
            "trial-{index}-{stem}",
            "--swanlab",
            "--swanlab-project",
            "YOLO",
            "--swanlab-run-name",
            "experiment",
            "--swanlab-mode",
            "offline",
            "--swanlab-log-interval",
            "3",
            "--extra",
            "optimizer=SGD",
            "amp=False",
        ]
    )
    cfg = train_auto.config_from_args(args, train_auto.AUTO_TRAIN_CONFIG)

    commands = train_auto.build_train_commands(cfg)

    assert "model=a.yaml" in commands[0]
    assert "name=trial-1-a" in commands[0]
    assert "model=b.yaml" in commands[1]
    assert "name=trial-2-b" in commands[1]
    assert "epochs=2" in commands[0]
    assert "batch=4" in commands[0]
    assert "imgsz=320" in commands[0]
    assert "swanlab=True" in commands[0]
    assert "swanlab_project=YOLO" in commands[0]
    assert "swanlab_run_name=experiment" in commands[0]
    assert "swanlab_mode=offline" in commands[0]
    assert "swanlab_log_interval=3" in commands[0]
    assert "optimizer=SGD" in commands[0]
    assert "amp=False" in commands[0]


def test_dry_run_does_not_launch_subprocess(monkeypatch, capsys):
    launched = []
    monkeypatch.setattr(train_auto.subprocess, "run", lambda *a, **k: launched.append((a, k)))

    code = train_auto.main(["--models", "a.yaml", "--data", "data.yaml", "--dry-run"])

    assert code == 0
    assert launched == []
    assert "DRY-RUN" in capsys.readouterr().out


def test_stop_on_fail_skips_remaining_jobs(monkeypatch):
    calls = []

    def fake_run(command, cwd, check):
        calls.append(command)
        return train_auto.subprocess.CompletedProcess(command, 3)

    monkeypatch.setattr(train_auto.subprocess, "run", fake_run)

    code = train_auto.main(["--models", "a.yaml", "b.yaml", "--data", "data.yaml", "--stop-on-fail"])

    assert code == 1
    assert len(calls) == 1


def test_train_auto_can_enable_baseline_monitor_from_python_config():
    cfg = train_auto.AutoTrainConfig(
        models=["a.yaml"],
        data="data.yaml",
        baseline_monitor=True,
        baseline_path="runs/baseline/results.csv",
        baseline_metric="metrics/mAP50-95(B)",
        baseline_start_ratio=0.4,
        baseline_min_gap=0.02,
        baseline_patience=2,
    )

    command = train_auto.build_train_commands(cfg)[0]

    assert "baseline_monitor=True" in command
    assert "baseline_path=runs/baseline/results.csv" in command
    assert "baseline_metric=metrics/mAP50-95(B)" in command
    assert "baseline_start_ratio=0.4" in command
    assert "baseline_min_gap=0.02" in command
    assert "baseline_patience=2" in command
