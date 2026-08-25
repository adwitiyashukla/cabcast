from __future__ import annotations

import json

import pytest

from cabcast.cli import main

pytestmark = pytest.mark.slow

TINY = [
    "data.start_month=2024-10",
    "data.end_month=2025-04",
    "data.synthetic_trips_per_month=45000",
    "geo.min_daily_trips=4",
    "split.n_folds=2",
    "split.test_days=5",
    "split.validation_days=4",
    "conformal.calibration_days=5",
    "conformal.mondrian_bins=3",
    "causal.event_window_months=3",
    "causal.min_weekly_trips=5",
    "model.lgbm.n_estimators=60",
    "model.lgbm.num_leaves=16",
]


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("cabcast_run")
    overrides = TINY + [
        f"paths.{name}={root / name}"
        for name in ("bronze", "silver", "gold", "external", "artifacts", "reports")
    ] + [f"paths.figures={root / 'reports' / 'figures'}"]

    code = main(["all", "--source", "synthetic", *[f"--set={o}" for o in overrides]])
    assert code == 0
    results = json.loads((root / "reports" / "results.json").read_text(encoding="utf-8"))
    return root, results


def test_pipeline_completes_and_writes_results(pipeline_run):
    root, results = pipeline_run
    assert results["n_rows_features"] > 0
    assert results["n_features"] > 40
    assert (root / "gold" / "demand_panel.parquet").exists()
    assert (root / "silver" / "trips.parquet").exists()


def test_every_declared_figure_exists_on_disk(pipeline_run):
    root, results = pipeline_run
    assert len(results["figures"]) >= 12
    for name in results["figures"].values():
        assert (root / "reports" / "figures" / name).stat().st_size > 5000


def test_models_are_registered_with_cards(pipeline_run):
    root, _ = pipeline_run
    for name in ("demand_point", "demand_q_lower", "demand_q_upper"):
        assert (root / "artifacts" / name / "model.txt").exists()
        card = json.loads((root / "artifacts" / name / "card.json").read_text(encoding="utf-8"))
        assert card["n_features"] > 0
        assert card["data_source"]
    assert (root / "artifacts" / "feature_store.parquet").exists()
    assert (root / "artifacts" / "conformal.json").exists()


def test_learned_model_beats_the_seasonal_naive_baseline(pipeline_run):
    _, results = pipeline_run
    point = results["test"]["point"]
    assert point["lightgbm"]["mae"] < point["seasonal_naive"]["mae"]


def test_conformal_intervals_land_near_nominal_coverage(pipeline_run):
    _, results = pipeline_run
    intervals = results["test"]["intervals"]
    target = intervals["mondrian_cqr"]["target_coverage"]
    assert abs(intervals["mondrian_cqr"]["coverage"] - target) < 0.06


def test_rebalancing_and_causal_blocks_are_populated(pipeline_run):
    _, results = pipeline_run
    reb = results["rebalancing"]
    assert reb["sinkhorn_converged"] is True
    assert reb["unmet_after"] <= reb["unmet_before"]
    assert 0.0 <= reb["unmet_reduction_pct"] <= 100.0
    assert len(reb["frontier"]) > 1

    causal = results["causal"]
    assert causal["status"] == "ok"
    assert causal["n_obs"] > 0
    assert causal["n_zones"] >= 6


def test_rebalancing_frontier_has_diminishing_returns(pipeline_run):
    import pandas as pd

    _, results = pipeline_run
    df = pd.DataFrame(results["rebalancing"]["frontier"])
    horizon = df["horizon_minutes"].mode().iloc[0]
    sub = df[df["horizon_minutes"] == horizon].sort_values("reposition_share")
    assert sub["unmet_reduction_pct"].is_monotonic_increasing
    assert sub["minutes_per_extra_trip"].iloc[-1] > sub["minutes_per_extra_trip"].iloc[0]


def test_quality_reports_are_written(pipeline_run):
    root, _ = pipeline_run
    silver = json.loads((root / "reports" / "quality_silver.json").read_text(encoding="utf-8"))
    assert silver["rows_in"] > silver["rows_out"]
    assert silver["quarantine_rate"] < 0.15
    assert (root / "reports" / "results.md").exists()
