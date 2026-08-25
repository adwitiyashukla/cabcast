from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cabcast.causal.did import (
    build_did_panel,
    estimate_did,
    event_study,
    panel_is_estimable,
    run_congestion_pricing_study,
)

TREATMENT = "2025-01-05"
TRUE_EFFECT = -0.12


def _panel(
    n_treated: int = 16,
    n_control: int = 24,
    weeks: int = 20,
    shock_sd: float = 0.06,
    seed: int = 3,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(TREATMENT)
    hours = pd.date_range(start - pd.Timedelta(weeks=weeks // 2), periods=weeks * 168, freq="h")
    post = np.asarray(hours >= start)
    week_index = np.arange(len(hours)) // 168

    rows = []
    for zone in range(1, n_treated + n_control + 1):
        shocks = rng.lognormal(0.0, shock_sd, week_index.max() + 1)[week_index]
        lam = np.full(len(hours), 40.0 + zone, dtype=float) * shocks
        treated = zone <= n_treated
        if treated:
            lam = lam * np.where(post, 1.0 + TRUE_EFFECT, 1.0)
        rows.append(
            pd.DataFrame(
                {
                    "zone_id": zone,
                    "hour_ts": hours,
                    "trips": rng.poisson(lam).astype(float),
                    "in_crz": treated,
                    "borough": "Manhattan",
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def agg():
    return build_did_panel(_panel(), TREATMENT, window_months=2, control_boroughs=["Manhattan"])


def test_panel_has_the_expected_columns(agg):
    for col in ("zone_id", "period", "trips", "treated", "post", "treat_post", "event_time"):
        assert col in agg.columns


def test_event_time_is_zero_at_the_treatment_period(agg):
    at_treatment = agg[agg["event_time"] == 0]["period"].unique()
    assert len(at_treatment) == 1
    assert pd.Timestamp(at_treatment[0]) >= pd.Timestamp(TREATMENT) - pd.Timedelta(days=7)


def test_event_time_is_an_integer_offset(agg):
    assert agg["event_time"].notna().all()
    assert pd.api.types.is_integer_dtype(agg["event_time"].astype(int))


def test_post_flag_aligns_with_event_time(agg):
    assert (agg.loc[agg["event_time"] >= 0, "post"] == 1).all()
    assert (agg.loc[agg["event_time"] < 0, "post"] == 0).all()


def test_estimator_recovers_a_planted_effect(agg):
    result = estimate_did(agg)
    truth = TRUE_EFFECT * 100.0
    assert result.status == "ok"
    assert result.ci_low_pct <= truth <= result.ci_high_pct
    assert abs(result.att_pct - truth) < 4.0


def test_no_pre_trend_when_none_was_planted(agg):
    _, pretrend_p = event_study(agg)
    assert pretrend_p > 0.05


def test_event_study_post_coefficients_are_negative(agg):
    table, _ = event_study(agg)
    post = table[table["event_time"] >= 0]
    assert post["coef_pct"].mean() < 0


def test_degenerate_panels_are_detected():
    tiny = build_did_panel(
        _panel(n_treated=1, n_control=1, weeks=8), TREATMENT, window_months=1,
        control_boroughs=["Manhattan"],
    )
    assert panel_is_estimable(tiny) is not None


def test_estimator_is_stable_across_seeds():
    estimates = []
    for seed in (11, 12, 13):
        agg = build_did_panel(
            _panel(seed=seed), TREATMENT, window_months=2, control_boroughs=["Manhattan"]
        )
        estimates.append(estimate_did(agg).att_pct)
    assert max(estimates) - min(estimates) < 4.0
    assert all(e < 0 for e in estimates)


def test_study_returns_a_labelled_null_instead_of_raising():
    class Cfg:
        class causal:
            treatment_start = TREATMENT
            event_window_months = 1

            @staticmethod
            def get(key, default=None):
                return {"control_boroughs": ["Manhattan"], "min_weekly_trips": 1e9}.get(key, default)

    result = run_congestion_pricing_study(_panel(n_treated=2, n_control=2, weeks=8), Cfg)
    assert result.status != "ok"
    assert np.isnan(result.att_pct)
    assert result.event_study.empty


def _panel_with_pretrend(drift_per_week: float = -0.012, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(TREATMENT)
    weeks = 20
    hours = pd.date_range(start - pd.Timedelta(weeks=weeks // 2), periods=weeks * 168, freq="h")
    post = np.asarray(hours >= start)
    week_index = np.arange(len(hours)) // 168
    centred = week_index - week_index.max() / 2.0

    rows = []
    for zone in range(1, 41):
        treated = zone <= 16
        shocks = rng.lognormal(0.0, 0.05, week_index.max() + 1)[week_index]
        lam = np.full(len(hours), 40.0 + zone, dtype=float) * shocks
        if treated:
            lam = lam * (1.0 + drift_per_week) ** centred
            lam = lam * np.where(post, 1.0 + TRUE_EFFECT, 1.0)
        rows.append(
            pd.DataFrame(
                {
                    "zone_id": zone,
                    "hour_ts": hours,
                    "trips": rng.poisson(np.clip(lam, 0.1, None)).astype(float),
                    "in_crz": treated,
                    "borough": "Manhattan",
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


class _Cfg:
    class causal:
        treatment_start = TREATMENT
        event_window_months = 2

        @staticmethod
        def get(key, default=None):
            return {"control_boroughs": ["Manhattan"], "min_weekly_trips": 1.0}.get(key, default)


def test_clean_panel_is_reported_as_parallel():
    result = run_congestion_pricing_study(_panel(), _Cfg)
    assert result.status == "ok"
    assert result.parallel_trends_holds is True
    assert result.robustness == {}


def test_planted_pretrend_is_detected_and_flagged():
    result = run_congestion_pricing_study(_panel_with_pretrend(), _Cfg)
    assert result.status == "ok"
    assert result.parallel_trends_holds is False
    assert result.pretrend_slope_pct_per_period < -0.3


def test_violation_triggers_a_zone_trend_robustness_estimate():
    result = run_congestion_pricing_study(_panel_with_pretrend(), _Cfg)
    adjusted = result.robustness.get("twoway_fe_zone_trends")
    assert adjusted is not None
    assert adjusted["spec"] == "twoway_fe_zone_trends"
    assert abs(adjusted["att_pct"] - TRUE_EFFECT * 100.0) < abs(
        result.att_pct - TRUE_EFFECT * 100.0
    )
