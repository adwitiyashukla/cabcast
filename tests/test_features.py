from __future__ import annotations

import numpy as np
import pytest

from urbanflow.features.build import LEAKY, assert_no_leakage, build_features


@pytest.fixture(scope="module")
def built(toy_panel, graph, cfg):
    small = toy_panel.copy()
    return build_features(small, graph, cfg)


def test_contemporaneous_columns_never_enter_features(built):
    _, cols = built
    assert_no_leakage(cols)
    for col in [*LEAKY, "trips"]:
        assert col not in cols


def test_leaky_columns_survive_only_as_lags(built):
    _, cols = built
    assert "mean_fare" not in cols
    assert "mean_fare_lag24" in cols


def test_lag_columns_match_a_manual_shift(built):
    df, _ = built
    zone = df[df["zone_id"] == df["zone_id"].iloc[0]].sort_values("hour_ts").reset_index(drop=True)
    manual = zone["trips"].shift(24)
    assert np.allclose(zone["lag_24h"][24:], manual[24:], equal_nan=True)


def test_rolling_mean_excludes_the_current_hour(built):
    df, _ = built
    zone = df[df["zone_id"] == df["zone_id"].iloc[0]].sort_values("hour_ts").reset_index(drop=True)
    window = 24
    row = 400
    expected = zone["trips"].iloc[row - window : row].mean()
    assert zone[f"roll_mean_{window}h"].iloc[row] == pytest.approx(expected, rel=1e-5)


def test_warmup_rows_are_dropped(toy_panel, built, cfg):
    df, _ = built
    span = (df["hour_ts"].min() - toy_panel["hour_ts"].min()).total_seconds() / 3600
    assert span >= max(cfg.features.lags_hours)


def test_fourier_terms_are_bounded(built):
    df, cols = built
    fourier = [c for c in cols if c.startswith("fourier_")]
    assert fourier
    values = df[fourier].to_numpy()
    assert np.nanmax(np.abs(values)) <= 1.0 + 1e-9


def test_spatial_features_present_and_finite(built):
    df, cols = built
    spatial = [c for c in cols if c.startswith("spatial_")]
    assert spatial
    assert np.isfinite(df[spatial].to_numpy()).any()


def test_leakage_guard_raises_on_a_bad_list():
    with pytest.raises(ValueError, match="leaked"):
        assert_no_leakage(["hour", "arrivals"])
