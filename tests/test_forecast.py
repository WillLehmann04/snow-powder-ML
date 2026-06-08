"""Smoke tests for forecast.py inference pipeline."""

import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

MODEL_PATH = Path("data/models/xgb_overnight_snow.pkl")


@pytest.fixture
def model_payload():
    if not MODEL_PATH.exists():
        pytest.skip("Model not trained yet — run: python -m core.train")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _fake_hourly(n_days: int = 7, seed: int = 0) -> pd.DataFrame:
    """Minimal hourly DataFrame mimicking Open-Meteo forecast response."""
    rng = np.random.default_rng(seed)
    n   = n_days * 24
    idx = pd.date_range("2025-12-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "temperature_2m":       rng.normal(-8, 5, n),
            "dewpoint_2m":          rng.normal(-12, 5, n),
            "snowfall":             np.clip(rng.exponential(0.5, n), 0, 10),
            "precipitation":        np.clip(rng.exponential(0.3, n), 0, 5),
            "wind_speed_10m":       np.clip(rng.exponential(15, n), 0, 80),
            "wind_direction_10m":   rng.uniform(0, 360, n),
            "relative_humidity_2m": np.clip(rng.normal(75, 10, n), 10, 100),
            "shortwave_radiation":  np.clip(rng.exponential(20, n), 0, 200),
            "cloud_cover":          np.clip(rng.normal(70, 20, n), 0, 100),
            "pressure_msl":         rng.normal(1005, 15, n),
            "surface_pressure":     rng.normal(895, 10, n),
            "temperature_850hPa":   rng.normal(-10, 4, n),
        },
        index=idx,
    )


def test_powder_score_zero_when_no_snow():
    from forecast import powder_score
    assert powder_score(0.0, -5.0, 10.0) == 0
    assert powder_score(0.4, -5.0, 10.0) == 0   # below 0.5cm threshold


def test_powder_score_in_range():
    from forecast import powder_score
    for snow in [0, 1, 5, 10, 15, 20, 30]:
        score = powder_score(float(snow), -8.0, 5.0)
        assert 0 <= score <= 100, f"Score {score} out of range for snow={snow}cm"


def test_powder_score_increases_with_snow():
    from forecast import powder_score
    scores = [powder_score(float(s), -8.0, 5.0) for s in [0, 5, 10, 15, 20]]
    assert scores == sorted(scores), "Powder score should increase with snowfall"


def test_japan_resort_forecast_uses_nonzero_amplification(model_payload):
    """Japan resort must receive its stored amplification factor, not the default 0."""
    from forecast import forecast_resort

    amp_per_resort = model_payload["nwp_amplification_per_resort"]
    model     = model_payload["model"]
    feat_cols = model_payload["feature_cols"]

    # Niseko should have amplification well above 1.0
    niseko_amp = amp_per_resort.get("niseko_grand_hirafu", 0)
    assert niseko_amp > 1.0, f"Niseko amplification in payload is {niseko_amp:.2f} (expect >1.0)"

    cfg = {
        "lat": 42.8643, "lon": 140.7009, "elevation": 1200,
        "region": "hokkaido", "hemisphere": "north",
    }

    with patch("forecast.fetch_hourly", return_value=_fake_hourly(7)):
        results = forecast_resort(
            "niseko_grand_hirafu", cfg, model, feat_cols,
            calibrations={}, amp_per_resort=amp_per_resort, days=7,
        )

    assert len(results) == 7
    assert "error" not in results[0], f"Forecast returned error: {results[0].get('error')}"
    for day in results:
        assert 0 <= day["powder_score"] <= 100, f"Score out of range: {day['powder_score']}"
        assert day["predicted_snow_cm"] >= 0,   f"Negative prediction: {day['predicted_snow_cm']}"


def test_sh_resort_forecast_runs(model_payload):
    """SH resort forecast (NWP-direct path) must complete without error."""
    from forecast import forecast_resort

    amp_per_resort = model_payload["nwp_amplification_per_resort"]
    model     = model_payload["model"]
    feat_cols = model_payload["feature_cols"]

    cfg = {
        "lat": -36.5054, "lon": 148.3009, "elevation": 2037,
        "region": "nsw", "hemisphere": "south",
    }

    with patch("forecast.fetch_hourly", return_value=_fake_hourly(7, seed=42)):
        results = forecast_resort(
            "thredbo", cfg, model, feat_cols,
            calibrations={}, amp_per_resort=amp_per_resort, days=7,
        )

    assert len(results) == 7
    assert "error" not in results[0], f"Forecast returned error: {results[0].get('error')}"
