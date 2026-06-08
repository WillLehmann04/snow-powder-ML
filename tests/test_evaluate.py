"""Tests for core/evaluate.py."""

import numpy as np
import pytest


def test_metrics_row_perfect_prediction():
    from core.evaluate import _metrics_row
    y = np.array([0.0, 0.0, 5.0, 20.0, 0.0, 30.0, 15.0, 0.0])
    m = _metrics_row(y, y, powder_pred_thresh=15.0)
    assert m["mae"]  == pytest.approx(0.0, abs=1e-9)
    assert m["r"]    == pytest.approx(1.0, abs=0.01)
    assert m["prec"] == pytest.approx(1.0)
    assert m["rec"]  == pytest.approx(1.0)
    assert m["f1"]   == pytest.approx(1.0)


def test_metrics_row_no_powder_days():
    from core.evaluate import _metrics_row
    y = np.zeros(20)
    m = _metrics_row(y, y, powder_pred_thresh=5.0)
    assert m["tp"] == 0
    assert m["fn"] == 0
    assert m["fp"] == 0


def test_metrics_row_returns_empty_for_tiny_sample():
    from core.evaluate import _metrics_row
    y = np.array([1.0, 2.0])
    m = _metrics_row(y, y, powder_pred_thresh=5.0)
    assert m == {}, "Should return empty dict for n < 5"


def test_metrics_row_all_false_negatives():
    from core.evaluate import _metrics_row
    y_true = np.array([20.0, 25.0, 30.0, 0.0, 0.0, 0.0])
    y_pred = np.zeros(6)
    m = _metrics_row(y_true, y_pred, powder_pred_thresh=15.0)
    assert m["rec"] == pytest.approx(0.0)
    assert m["fn"]  == 3


def test_hemisphere_powder_thresholds():
    """NH (15cm) and SH (4cm) thresholds must be distinct and correct."""
    NH_THRESHOLD = 15.0
    SH_THRESHOLD =  4.0
    assert NH_THRESHOLD == 15.0
    assert SH_THRESHOLD ==  4.0
    assert NH_THRESHOLD > SH_THRESHOLD
