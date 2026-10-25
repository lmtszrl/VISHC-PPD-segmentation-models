import numpy as np
import pytest

from vishc_ppd.metrics import compute_metrics, per_image_iou


def test_background_false_positives_are_penalised():
    gt = np.zeros((10, 10), int)
    gt[2:6, 2:6] = 1                       # 16 px of P1
    pred = gt.copy()
    pred[6:8, 2:6] = 1                     # 8 extra px predicted on background
    m = compute_metrics(pred, gt)
    assert m["iou_P1"] == pytest.approx(16 / 24)
    assert m["precision_P1"] == pytest.approx(16 / 24)
    assert m["acc_P1"] == pytest.approx(1.0)


def test_perfect_prediction():
    gt = np.random.RandomState(0).randint(0, 9, size=(32, 32))
    m = compute_metrics(gt, gt)
    assert m["miou"] == pytest.approx(1.0) and m["f1"] == pytest.approx(1.0) and m["pixel_acc"] == pytest.approx(1.0)


def test_absent_class_excluded_from_macro_average():
    gt = np.zeros((10, 10), int)
    gt[:5] = 1
    pred = gt.copy()
    m = compute_metrics(pred, gt)          # only P1 present -> macro over P1 only
    assert m["miou"] == pytest.approx(1.0)
    assert per_image_iou(pred, gt)["iou_P2"] is None
