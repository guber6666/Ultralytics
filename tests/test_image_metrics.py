import numpy as np

from ultralytics.utils.metrics import DetMetrics, Metric, PoseMetrics, SegmentMetrics


def test_metric_update_image_metrics_uses_iou50_column():
    metric = Metric()
    tp = np.array([[True, False], [False, True], [True, True]], dtype=bool)
    target_cls = np.array([0, 1, 1], dtype=float)
    pred_cls = np.array([0, 1, 1], dtype=float)

    metric.update_image_metrics(tp, target_cls, pred_cls, "im.jpg")

    assert metric.image_metrics["im.jpg"] == {
        "precision": 2 / 3,
        "recall": 2 / 3,
        "f1": 2 / 3,
        "tp": 2,
        "fp": 1,
        "fn": 1,
    }


def test_det_metrics_update_stats_records_box_image_metrics():
    metrics = DetMetrics()
    stat = {
        "tp": np.array([[True] + [False] * 9, [False] * 10], dtype=bool),
        "conf": np.array([0.9, 0.2], dtype=float),
        "pred_cls": np.array([0, 1], dtype=float),
        "target_cls": np.array([0], dtype=float),
        "target_img": np.array([0], dtype=float),
        "im_name": "det.jpg",
    }

    metrics.update_stats(stat)

    assert metrics.box.image_metrics["det.jpg"] == {
        "precision": 0.5,
        "recall": 1.0,
        "f1": 2 / 3,
        "tp": 1,
        "fp": 1,
        "fn": 0,
    }


def test_segment_metrics_update_stats_records_box_and_mask_image_metrics():
    metrics = SegmentMetrics()
    stat = {
        "tp": np.array([[True] + [False] * 9], dtype=bool),
        "tp_m": np.array([[False] * 10], dtype=bool),
        "conf": np.array([0.9], dtype=float),
        "pred_cls": np.array([0], dtype=float),
        "target_cls": np.array([0], dtype=float),
        "target_img": np.array([0], dtype=float),
        "im_name": "seg.jpg",
    }

    metrics.update_stats(stat)

    assert metrics.box.image_metrics["seg.jpg"]["tp"] == 1
    assert metrics.seg.image_metrics["seg.jpg"] == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "tp": 0,
        "fp": 1,
        "fn": 1,
    }


def test_pose_metrics_update_stats_records_box_and_pose_image_metrics():
    metrics = PoseMetrics()
    stat = {
        "tp": np.array([[True] + [False] * 9], dtype=bool),
        "tp_p": np.array([[False] * 10], dtype=bool),
        "conf": np.array([0.9], dtype=float),
        "pred_cls": np.array([0], dtype=float),
        "target_cls": np.array([0], dtype=float),
        "target_img": np.array([0], dtype=float),
        "im_name": "pose.jpg",
    }

    metrics.update_stats(stat)

    assert metrics.box.image_metrics["pose.jpg"]["tp"] == 1
    assert metrics.pose.image_metrics["pose.jpg"] == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "tp": 0,
        "fp": 1,
        "fn": 1,
    }
