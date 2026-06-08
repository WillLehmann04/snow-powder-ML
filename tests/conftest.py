"""Shared fixtures for the test suite."""

import numpy as np
import pandas as pd
import pytest


def _make_hourly(
    start: str = "2023-11-01",
    n_days: int = 30,
    seed: int = 42,
    temp_mean: float = -5.0,
) -> pd.DataFrame:
    """Create a minimal hourly weather DataFrame suitable for build_features()."""
    rng = np.random.default_rng(seed)
    n   = n_days * 24
    idx = pd.date_range(start, periods=n, freq="h")
    return pd.DataFrame(
        {
            "temperature_2m":       rng.normal(temp_mean, 8, n),
            "dewpoint_2m":          rng.normal(temp_mean - 4, 6, n),
            "snowfall":             np.clip(rng.exponential(0.3, n), 0, 10),
            "precipitation":        np.clip(rng.exponential(0.2, n), 0, 5),
            "wind_speed_10m":       np.clip(rng.exponential(15, n), 0, 80),
            "wind_direction_10m":   rng.uniform(0, 360, n),
            "relative_humidity_2m": np.clip(rng.normal(70, 15, n), 10, 100),
            "shortwave_radiation":  np.clip(rng.exponential(50, n), 0, 600),
            "cloud_cover":          np.clip(rng.normal(60, 30, n), 0, 100),
            "pressure_msl":         rng.normal(1013, 10, n),
            "surface_pressure":     rng.normal(900, 10, n),
            "temperature_850hPa":   rng.normal(-8, 5, n),
        },
        index=idx,
    )


@pytest.fixture
def hourly_nh() -> pd.DataFrame:
    """30 days of northern-hemisphere hourly weather (Nov)."""
    return _make_hourly(start="2023-11-01", n_days=30)


@pytest.fixture
def hourly_sh() -> pd.DataFrame:
    """30 days of southern-hemisphere hourly weather (Jul)."""
    return _make_hourly(start="2023-07-01", n_days=30, seed=99, temp_mean=-2.0)


@pytest.fixture
def hourly_two_seasons() -> pd.DataFrame:
    """10 days spanning the Sep→Oct NH season boundary (Sep 28 – Oct 6).

    NH season_year: month >= 10 → same year, month < 10 → year - 1.
    Sep 30 is season 2021; Oct 1 is the first day of season 2022.
    """
    return _make_hourly(start="2022-09-28", n_days=10, seed=7)
