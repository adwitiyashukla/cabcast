from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from urbanflow.causal.did import run_congestion_pricing_study
from urbanflow.evaluation.metrics import coverage
from urbanflow.geo.graph import ZoneGraph
from urbanflow.geo.zones import ZoneSet
from urbanflow.logging_utils import get_logger, log_event, stage
from urbanflow.models.conformal import (
    MondrianConformalizedQuantile,
    SplitConformal,
)
from urbanflow.monitoring.drift import feature_drift, prediction_drift
from urbanflow.optimize.rebalance import evaluate_rebalancing
from urbanflow.pipelines.train import TrainingArtifacts
from urbanflow.viz import plots

log = get_logger(__name__)

ALPHA_GRID = [0.30, 0.20, 0.10, 0.05, 0.02]


def build_test_frame(artifacts: TrainingArtifacts) -> pd.DataFrame:
    frame = pd.DataFrame(artifacts.results["predictions"])
    frame["hour_ts"] = pd.to_datetime(frame["hour_ts"])
    return frame


def coverage_curves(artifacts: TrainingArtifacts, frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cal = artifacts.calibration
    curves: dict[str, list[dict[str, float]]] = {
        "quantile_uncalibrated": [],
        "split_conformal": [],
        "mondrian_cqr": [],
    }
    y = frame["y"].to_numpy()
    active = y > 0

    for alpha in ALPHA_GRID:
        target = 1.0 - alpha
        curves["quantile_uncalibrated"].append(
            {
                "target": target,
                "empirical": coverage(y, frame["q_lower"], frame["q_upper"]),
                "empirical_active": coverage(
                    y[active], frame["q_lower"].to_numpy()[active],
                    frame["q_upper"].to_numpy()[active],
                ),
            }
        )
        sc = SplitConformal(alpha).calibrate(cal["y"], cal["point"])
        lo, hi = sc.interval(frame["yhat"].to_numpy())
        curves["split_conformal"].append(
            {
                "target": target,
                "empirical": coverage(y, lo, hi),
                "empirical_active": coverage(y[active], lo[active], hi[active]),
            }
        )

        mc = MondrianConformalizedQuantile(alpha, 6).calibrate(
            cal["y"], cal["lower"], cal["upper"], cal["point"]
        )
        lo, hi = mc.interval(
            frame["q_lower"].to_numpy(), frame["q_upper"].to_numpy(), frame["yhat"].to_numpy()
        )
        curves["mondrian_cqr"].append(
            {
                "target": target,
                "empirical": coverage(y, lo, hi),
                "empirical_active": coverage(y[active], lo[active], hi[active]),
            }
        )

    return {k: pd.DataFrame(v) for k, v in curves.items()}


def run_rebalancing(frame: pd.DataFrame, graph: ZoneGraph, cfg) -> dict[str, Any]:
    hourly = frame.groupby("hour_ts", observed=True)["yhat"].sum()
    peak_hour = hourly.idxmax()
    snapshot = frame[frame["hour_ts"] == peak_hour].sort_values("zone_id")

    zone_ids = snapshot["zone_id"].to_numpy()
    idx = np.searchsorted(graph.zone_ids, zone_ids)
    travel = graph.travel_minutes[np.ix_(idx, idx)]

    predicted = snapshot["yhat"].to_numpy()
    rng = np.random.default_rng(int(cfg.project.random_seed))
    baseline = frame.groupby("zone_id", observed=True)["y"].mean().reindex(zone_ids).to_numpy()
    idle_supply = np.maximum(baseline * rng.lognormal(0.0, 0.45, len(baseline)), 1e-3)

    result, plan = evaluate_rebalancing(
        idle_supply=idle_supply,
        predicted_demand=predicted,
        travel_minutes=travel,
        fleet_size=float(cfg.optimize.idle_fleet_size),
        service_radius_minutes=float(cfg.optimize.service_radius_minutes),
        epsilon=float(cfg.optimize.sinkhorn_epsilon),
        max_iter=int(cfg.optimize.sinkhorn_max_iter),
        tol=float(cfg.optimize.sinkhorn_tol),
        reposition_share=float(cfg.optimize.get("reposition_share", 0.35)),
        horizon_minutes=float(cfg.optimize.get("arrival_horizon_minutes", 30.0)),
    )
    frontier = []
    for share in list(cfg.optimize.get("frontier_shares", [0.25])):
        for horizon in list(cfg.optimize.get("frontier_horizons", [30.0])):
            fr, _ = evaluate_rebalancing(
                idle_supply=idle_supply, predicted_demand=predicted, travel_minutes=travel,
                fleet_size=float(cfg.optimize.idle_fleet_size),
                service_radius_minutes=float(cfg.optimize.service_radius_minutes),
                epsilon=float(cfg.optimize.sinkhorn_epsilon),
                max_iter=int(cfg.optimize.sinkhorn_max_iter),
                tol=float(cfg.optimize.sinkhorn_tol),
                reposition_share=float(share), horizon_minutes=float(horizon),
            )
            frontier.append(
                {
                    "reposition_share": float(share),
                    "horizon_minutes": float(horizon),
                    "unmet_reduction_pct": round(fr.unmet_reduction_pct, 3),
                    "minutes_per_extra_trip": round(fr.minutes_per_extra_trip, 3),
                    "vehicles_moved": round(fr.vehicles_moved, 1),
                    "vehicles_stranded": round(fr.vehicles_stranded, 1),
                }
            )

    return {
        "peak_hour": str(peak_hour),
        "frontier": frontier,
        "reposition_share": float(cfg.optimize.get("reposition_share", 0.35)),
        "arrival_horizon_minutes": float(cfg.optimize.get("arrival_horizon_minutes", 30.0)),
        "vehicles_moved": round(result.vehicles_moved, 1),
        "vehicles_stranded": round(result.vehicles_stranded, 1),
        "n_zones": int(len(zone_ids)),
        "fleet_size": float(cfg.optimize.idle_fleet_size),
        "unmet_before": round(result.unmet_before, 2),
        "unmet_after": round(result.unmet_after, 2),
        "unmet_reduction_pct": round(result.unmet_reduction_pct, 2),
        "served_before": round(result.served_before, 2),
        "served_after": round(result.served_after, 2),
        "vehicle_minutes": round(result.vehicle_minutes, 1),
        "minutes_per_extra_trip": round(result.minutes_per_extra_trip, 2),
        "sinkhorn_iterations": plan.iterations,
        "sinkhorn_converged": bool(plan.converged),
        "sinkhorn_marginal_error": float(plan.marginal_error),
        "_moves": result.moves,
        "_zone_ids": zone_ids,
        "_supply": idle_supply,
        "_demand": predicted,
    }


def run_drift(artifacts: TrainingArtifacts, cfg) -> dict[str, Any]:
    ref = artifacts.drift_reference
    cur = pd.DataFrame(artifacts.test_x, columns=artifacts.feature_cols)
    report = feature_drift(
        ref, cur, artifacts.feature_cols,
        float(cfg.monitoring.psi_warn), float(cfg.monitoring.psi_alert),
        float(cfg.monitoring.ks_alpha),
    )
    pred = prediction_drift(
        artifacts.point_model.predict(ref.to_numpy(dtype=np.float32)),
        artifacts.point_model.predict(artifacts.test_x),
        float(cfg.monitoring.psi_alert),
    )
    return {"features": report.to_dict(), "prediction": pred, "_table": report.table}


def generate(
    artifacts: TrainingArtifacts,
    panel: pd.DataFrame,
    zones: ZoneSet,
    graph: ZoneGraph,
    cfg,
) -> dict[str, Any]:
    figures = cfg.path("figures")
    made: dict[str, str] = {}

    with stage(log, "reporting"):
        frame = build_test_frame(artifacts)

        made["demand_choropleth"] = str(
            plots.demand_choropleth(zones.gdf, panel, figures / "demand_choropleth.png").name
        )
        made["demand_heatmap"] = str(
            plots.demand_heatmap(panel, figures / "demand_heatmap.png").name
        )
        made["zone_network"] = str(
            plots.zone_network(zones.gdf, graph, figures / "zone_network.png").name
        )
        made["forecast_intervals"] = str(
            plots.forecast_intervals(frame, figures / "forecast_intervals.png").name
        )
        made["model_comparison"] = str(
            plots.model_comparison(
                artifacts.results["test"]["point"], figures / "model_comparison.png"
            ).name
        )
        made["backtest_folds"] = str(
            plots.backtest_folds(
                artifacts.results["backtest"]["per_fold"], figures / "backtest_folds.png"
            ).name
        )

        curves = coverage_curves(artifacts, frame)
        made["coverage_calibration"] = str(
            plots.coverage_calibration(curves, figures / "coverage_calibration.png").name
        )
        made["conditional_coverage"] = str(
            plots.conditional_coverage(
                artifacts.results["test"]["conditional_coverage"],
                1.0 - float(cfg.conformal.alpha),
                figures / "conditional_coverage.png",
            ).name
        )
        made["feature_importance"] = str(
            plots.feature_importance(
                artifacts.results["feature_importance"], figures / "feature_importance.png"
            ).name
        )
        made["residual_diagnostics"] = str(
            plots.residual_diagnostics(frame, figures / "residual_diagnostics.png").name
        )

        rebalance = run_rebalancing(frame, graph, cfg)
        made["rebalance_frontier"] = str(
            plots.rebalance_frontier(
                rebalance["frontier"], rebalance, figures / "rebalance_frontier.png"
            ).name
        )
        made["rebalancing_map"] = str(
            plots.rebalancing_map(
                zones.gdf, rebalance.pop("_zone_ids"), rebalance.pop("_moves"),
                rebalance.pop("_supply"), rebalance.pop("_demand"),
                figures / "rebalancing_map.png",
            ).name
        )

        drift = run_drift(artifacts, cfg)
        made["drift_panel"] = str(
            plots.drift_panel(
                drift.pop("_table"), float(cfg.monitoring.psi_warn),
                float(cfg.monitoring.psi_alert), figures / "drift_panel.png",
            ).name
        )

        causal = run_congestion_pricing_study(panel, cfg)
        estimable = causal.status == "ok"
        ground_truth = None
        if estimable and str(artifacts.results.get("data_source", "")).startswith("synthetic"):
            from urbanflow.data.synth import CRZ_SPILLOVER_EFFECT, CRZ_TRUE_EFFECT

            controls = list(cfg.causal.get("control_boroughs", ["Manhattan"]))
            spillover = CRZ_SPILLOVER_EFFECT if controls == ["Manhattan"] else 0.0
            estimand = ((1.0 + CRZ_TRUE_EFFECT) / (1.0 + spillover) - 1.0) * 100.0
            ground_truth = {
                "planted_effect_pct": round(estimand, 3),
                "recovered_in_ci": bool(
                    causal.ci_low_pct <= estimand <= causal.ci_high_pct
                ),
                "absolute_error_pct": round(abs(causal.att_pct - estimand), 3),
            }
        if estimable:
            made["event_study"] = str(
                plots.event_study(
                    causal.event_study,
                    figures / "event_study.png",
                    parallel_trends_holds=causal.parallel_trends_holds,
                    pretrend_p=causal.pretrend_p_value,
                    pretrend_slope=causal.pretrend_slope_pct_per_period,
                ).name
            )

        extra = {
            "figures": made,
            "rebalancing": rebalance,
            "drift": drift,
            "causal": {
                **causal.to_dict(),
                "control_boroughs": list(cfg.causal.get("control_boroughs", ["Manhattan"])),
                "ground_truth": ground_truth,
                "robustness": causal.robustness,
                "event_study": causal.event_study.round(5).to_dict(orient="records"),
            },
            "coverage_curves": {k: v.round(5).to_dict(orient="records") for k, v in curves.items()},
        }

    artifacts.results.update(extra)
    payload = {k: v for k, v in artifacts.results.items() if k != "predictions"}
    results_path = cfg.path("reports") / "results.json"
    results_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    frame.to_parquet(cfg.path("reports") / "test_predictions.parquet", index=False)
    log_event(log, "report complete", figures=len(made), path=str(results_path))
    return artifacts.results
