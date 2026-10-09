import csv

import pytest

from ultralytics.engine.baseline_log_monitor import BaselineLogMonitor


def _write_results_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_monitor_waits_until_start_ratio(tmp_path):
    baseline = tmp_path / "results.csv"
    _write_results_csv(
        baseline,
        [
            {"epoch": 1, "metrics/mAP50-95(B)": 0.20},
            {"epoch": 2, "metrics/mAP50-95(B)": 0.30},
        ],
    )
    monitor = BaselineLogMonitor.from_args(
        enabled=True,
        path=str(baseline),
        metric="metrics/mAP50-95(B)",
        total_epochs=4,
        start_ratio=0.5,
        min_gap=0.03,
        patience=1,
    )

    decision = monitor.check({"metrics/mAP50-95(B)": 0.10}, epoch=1)

    assert decision.active is False
    assert decision.should_stop is False
    assert decision.current == pytest.approx(0.10)
    assert decision.baseline == pytest.approx(0.20)


def test_monitor_stops_after_patience_bad_epochs(tmp_path):
    baseline = tmp_path / "results.csv"
    _write_results_csv(
        baseline,
        [
            {"epoch": 1, "metrics/mAP50-95(B)": 0.30},
            {"epoch": 2, "metrics/mAP50-95(B)": 0.40},
        ],
    )
    monitor = BaselineLogMonitor.from_args(True, str(baseline), "metrics/mAP50-95(B)", 2, 0.0, 0.03, 2)

    first = monitor.check({"metrics/mAP50-95(B)": 0.20}, epoch=1)
    second = monitor.check({"metrics/mAP50-95(B)": 0.30}, epoch=2)

    assert first.should_stop is False
    assert first.bad_epochs == 1
    assert second.should_stop is True
    assert second.bad_epochs == 2


def test_monitor_resets_bad_epochs_when_gap_recovers(tmp_path):
    baseline = tmp_path / "results.csv"
    _write_results_csv(
        baseline,
        [
            {"epoch": 1, "metrics/mAP50-95(B)": 0.30},
            {"epoch": 2, "metrics/mAP50-95(B)": 0.40},
        ],
    )
    monitor = BaselineLogMonitor.from_args(True, str(baseline), "metrics/mAP50-95(B)", 2, 0.0, 0.03, 2)

    bad = monitor.check({"metrics/mAP50-95(B)": 0.20}, epoch=1)
    recovered = monitor.check({"metrics/mAP50-95(B)": 0.38}, epoch=2)

    assert bad.bad_epochs == 1
    assert recovered.should_stop is False
    assert recovered.bad_epochs == 0


def test_monitor_does_not_stop_when_baseline_epoch_is_missing(tmp_path):
    baseline = tmp_path / "results.csv"
    _write_results_csv(baseline, [{"epoch": 1, "metrics/mAP50-95(B)": 0.30}])
    monitor = BaselineLogMonitor.from_args(True, str(baseline), "metrics/mAP50-95(B)", 2, 0.0, 0.03, 1)

    decision = monitor.check({"metrics/mAP50-95(B)": 0.10}, epoch=2)

    assert decision.active is True
    assert decision.should_stop is False
    assert decision.reason == "baseline epoch is missing"


def test_monitor_does_not_stop_when_current_metric_is_missing(tmp_path):
    baseline = tmp_path / "results.csv"
    _write_results_csv(baseline, [{"epoch": 1, "metrics/mAP50-95(B)": 0.30}])
    monitor = BaselineLogMonitor.from_args(True, str(baseline), "metrics/mAP50-95(B)", 1, 0.0, 0.03, 1)

    decision = monitor.check({}, epoch=1)

    assert decision.should_stop is False
    assert decision.reason == "metric is missing"


def test_monitor_missing_path_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="Baseline results.csv does not exist"):
        BaselineLogMonitor.from_args(True, str(tmp_path / "missing.csv"), "metrics/mAP50-95(B)", 1, 0.0, 0.03, 1)
