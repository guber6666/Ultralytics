from ultralytics.cfg import get_cfg


def test_baseline_monitor_args_are_valid_yolo_cfg_keys():
    cfg = get_cfg(
        overrides={
            "baseline_monitor": True,
            "baseline_path": "runs/baseline/results.csv",
            "baseline_metric": "metrics/mAP50-95(B)",
            "baseline_start_ratio": 0.5,
            "baseline_min_gap": 0.03,
            "baseline_patience": 2,
        }
    )

    assert cfg.baseline_monitor is True
    assert cfg.baseline_path == "runs/baseline/results.csv"
    assert cfg.baseline_metric == "metrics/mAP50-95(B)"
