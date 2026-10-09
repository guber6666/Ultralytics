from types import SimpleNamespace

from ultralytics.engine.trainer import BaseTrainer


def test_baseline_monitor_disabled_by_default():
    trainer = object.__new__(BaseTrainer)
    trainer.args = SimpleNamespace(baseline_monitor=False)
    trainer.epochs = 100

    trainer._build_baseline_log_monitor()

    assert trainer.baseline_log_monitor is None


def test_baseline_monitor_sets_stop_when_decision_requests_stop():
    trainer = object.__new__(BaseTrainer)
    trainer.stop = False
    trainer.epoch = 4
    trainer.metrics = {"metrics/mAP50-95(B)": 0.20}

    class FakeMonitor:
        min_gap = 0.03

        def check(self, metrics, epoch):
            return SimpleNamespace(
                active=True,
                should_stop=True,
                epoch=epoch,
                metric="metrics/mAP50-95(B)",
                current=0.20,
                baseline=0.30,
                diff=-0.10,
                bad_epochs=1,
                patience=1,
                reason="too weak",
            )

    trainer.baseline_log_monitor = FakeMonitor()

    decision = trainer._check_baseline_log_monitor()

    assert decision.should_stop is True
    assert trainer.stop is True
