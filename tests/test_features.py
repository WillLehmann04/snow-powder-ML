"""Tests for core/features.py."""

import numpy as np
import pandas as pd
import pytest

from core.features import build_features


def test_build_features_returns_daily_rows(hourly_nh):
    daily = build_features(hourly_nh, hemisphere="north")
    assert len(daily) == 30
    assert pd.api.types.is_datetime64_any_dtype(daily.index)


def test_expected_columns_present(hourly_nh):
    daily = build_features(hourly_nh, hemisphere="north")
    required = [
        "snowfall_24h", "snowfall_48h", "snowfall_72h", "snowfall_7d", "snowfall_14d",
        "temp_min", "temp_max", "wind_max", "humidity_mean",
        "pressure_mean", "pressure_tendency_24h",
        "dewpoint_depression", "snow_quality_index", "cold_air_advection",
        "day_of_year", "month", "season_year", "days_in_season",
    ]
    for col in required:
        assert col in daily.columns, f"Missing column: {col}"


def test_rolling_windows_reset_at_season_boundary(hourly_two_seasons):
    """snowfall_72h on Oct 1 must not include Sep data from a different season.

    NH boundary: Sep (month < 10, season year-1) → Oct (month >= 10, season year).
    Oct 1 is the first day of NH season 2022, so snowfall_72h must equal snowfall_24h.
    """
    daily = build_features(hourly_two_seasons, hemisphere="north")

    oct1 = pd.Timestamp("2022-10-01")
    if oct1 not in daily.index:
        pytest.skip("Oct 1 not in fixture date range")

    oct1_24h = daily.loc[oct1, "snowfall_24h"]
    oct1_72h = daily.loc[oct1, "snowfall_72h"]
    assert oct1_72h == pytest.approx(oct1_24h, rel=1e-6), (
        f"Rolling window crossed season boundary: "
        f"snowfall_72h={oct1_72h:.3f} != snowfall_24h={oct1_24h:.3f}"
    )


def test_no_negative_snowfall_features(hourly_nh):
    daily = build_features(hourly_nh, hemisphere="north")
    for col in ["snowfall_24h", "snowfall_48h", "snowfall_72h", "snowfall_7d"]:
        assert (daily[col] >= 0).all(), f"{col} has negative values"


def test_cold_air_advection_nonnegative(hourly_nh):
    daily = build_features(hourly_nh, hemisphere="north")
    assert (daily["cold_air_advection"] >= 0).all()


def test_cold_air_advection_differs_by_hemisphere(hourly_sh):
    """SH and NH cold_air_advection must differ (different wind components used)."""
    sh = build_features(hourly_sh, hemisphere="south")
    nh = build_features(hourly_sh, hemisphere="north")
    # They use wind_v vs wind_u respectively — values should differ
    assert not sh["cold_air_advection"].equals(nh["cold_air_advection"]), (
        "SH and NH cold_air_advection are identical — hemisphere branching is broken"
    )


def test_no_lag_features_in_output(hourly_nh):
    """build_features must not produce lag features — those belong in dataset.py."""
    daily = build_features(hourly_nh, hemisphere="north")
    lag_cols = {"overnight_snow_lag1", "overnight_snow_lag2",
                "snow_depth_lag1", "overnight_snow_3d_sum"}
    found = lag_cols & set(daily.columns)
    assert not found, f"Lag features must not appear in features.py output: {found}"


def test_season_year_assigned_correctly(hourly_nh):
    """NH: Oct–Dec → same year, Jan–Sep → year-1."""
    daily = build_features(hourly_nh, hemisphere="north")
    # Fixture starts 2023-11-01, so season_year should be 2023
    assert (daily["season_year"] == 2023).all(), "season_year wrong for Nov fixture"


def test_sh_season_year(hourly_sh):
    """SH: Apr–Dec → same year, Jan–Mar → year-1."""
    daily = build_features(hourly_sh, hemisphere="south")
    # Fixture starts 2023-07-01, so season_year should be 2023
    assert (daily["season_year"] == 2023).all(), "SH season_year wrong for Jul fixture"
