"""Tests for core/train.py — payload structure and feature list integrity."""

import pickle
from pathlib import Path

import pytest

MODEL_PATH = Path("data/models/xgb_overnight_snow.pkl")


@pytest.fixture
def payload():
    if not MODEL_PATH.exists():
        pytest.skip("Model not trained yet — run: python -m core.train")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def test_payload_has_required_keys(payload):
    required = {
        "model", "feature_cols", "holdout_season",
        "train_metrics", "test_metrics", "log_transform",
        "powder_pred_threshold", "nwp_amplification_per_resort", "trained_at",
    }
    missing = required - set(payload.keys())
    assert not missing, f"Payload missing keys: {missing}"


def test_nwp_amplification_per_resort_has_japan_resorts(payload):
    amp = payload["nwp_amplification_per_resort"]
    for resort in ["niseko_grand_hirafu", "niseko_annupuri", "kiroro", "rusutsu", "furano"]:
        assert resort in amp, f"Japan resort '{resort}' missing from nwp_amplification_per_resort"
        assert amp[resort] > 1.0, f"{resort} amplification {amp[resort]:.2f} should be > 1.0"
        assert amp[resort] <= 6.0, f"{resort} amplification {amp[resort]:.2f} exceeds 6x cap"


def test_no_lag_features_in_feature_cols(payload):
    lag = {"overnight_snow_lag1", "overnight_snow_lag2",
           "snow_depth_lag1", "overnight_snow_3d_sum"}
    found = lag & set(payload["feature_cols"])
    assert not found, f"Lag features must not be in FEATURE_COLS: {found}"


def test_model_test_metrics_reasonable(payload):
    m = payload["test_metrics"]
    assert m["r"] > 0.60,         f"Test r={m['r']:.3f} is too low (expect >0.60)"
    assert m["mae"] < 5.0,        f"Test MAE={m['mae']:.2f}cm is too high (expect <5.0cm)"
    assert m["powder_f1"] > 0.35, f"Powder F1={m['powder_f1']:.3f} is too low (expect >0.35)"


def test_log_transform_is_false(payload):
    """Tweedie model does not use a log transform — predictions are raw cm."""
    assert payload["log_transform"] is False


def test_trained_at_is_present(payload):
    assert "trained_at" in payload
    assert len(payload["trained_at"]) > 0
