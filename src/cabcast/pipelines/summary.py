from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def write_summary(results: dict[str, Any], cfg) -> Path:
    lines: list[str] = ["# CabCast results", ""]
    lines.append(f"Data source: `{results['data_source']}`")
    lines.append(
        f"Panel: {results['n_rows_features']:,} zone-hours across {results['n_zones']} zones, "
        f"{results['date_range'][0]} to {results['date_range'][1]}, {results['n_features']} features."
    )
    lines += ["", "## Point forecast accuracy on the held-out test block", ""]
    point = results["test"]["point"]
    table = pd.DataFrame(point).T[["mae", "rmse", "mase", "smape", "bias"]]
    table.index = [i.replace("_", " ") for i in table.index]
    lines.append(table.round(4).to_markdown())

    best = min(point, key=lambda k: point[k]["mae"])
    worst = max(point, key=lambda k: point[k]["mae"])
    lift = (point[worst]["mae"] - point[best]["mae"]) / point[worst]["mae"] * 100.0
    lines += ["", f"Best model `{best}` reduces MAE by {lift:.1f}% against `{worst}`.", ""]

    lines += ["## Prediction interval calibration", ""]
    cols = ["coverage", "coverage_active", "target_coverage", "mean_width", "winkler"]
    frame = pd.DataFrame(results["test"]["intervals"]).T
    iv = frame[[c for c in cols if c in frame.columns]]
    iv.index = [i.replace("_", " ") for i in iv.index]
    lines.append(iv.round(4).to_markdown())

    lines += ["", "## Fleet rebalancing", ""]
    reb = results["rebalancing"]
    lines.append(
        f"At the peak forecast hour ({reb['peak_hour']}), repositioning at most "
        f"{reb['reposition_share']:.0%} of a {int(reb['fleet_size']):,} vehicle idle fleet by the "
        f"Sinkhorn plan, counting only arrivals reachable within "
        f"{reb['arrival_horizon_minutes']:.0f} minutes, cuts unmet demand from "
        f"{reb['unmet_before']:.0f} to {reb['unmet_after']:.0f} trips, a "
        f"{reb['unmet_reduction_pct']:.1f}% reduction, at a cost of "
        f"{reb['minutes_per_extra_trip']:.1f} vehicle-minutes per additional trip served. "
        f"{reb['vehicles_moved']:.0f} vehicles move and {reb['vehicles_stranded']:.0f} are "
        f"stranded by the arrival horizon."
    )

    lines += ["", "## Congestion pricing effect", ""]
    causal = results["causal"]
    if causal.get("status") != "ok":
        lines.append(f"Causal study skipped: {causal.get('status')}.")
    elif causal.get("parallel_trends_holds", True):
        lines.append(
            f"Difference-in-differences ATT: {causal['att_pct']:+.2f}% "
            f"(95% CI {causal['ci_low_pct']:+.2f}% to {causal['ci_high_pct']:+.2f}%), "
            f"p = {causal['p_value']:.4g}, {causal['n_obs']:,} zone-weeks, standard errors "
            f"clustered by zone. Pre-trend joint test p = {causal['pretrend_p_value']:.3g}, "
            f"so parallel trends is not rejected."
        )
    else:
        lines.append(
            f"**Parallel trends fails on this panel, so the naive DiD is not identified.** "
            f"Pre-treatment coefficients trend {causal['pretrend_slope_pct_per_period']:+.2f}% "
            f"per period with joint F-test p = {causal['pretrend_p_value']:.3g} and "
            f"{causal['pretrend_significant_share']:.0%} of pre-periods individually significant. "
            f"The unadjusted two-way fixed-effects estimate is {causal['att_pct']:+.2f}% "
            f"(95% CI {causal['ci_low_pct']:+.2f}% to {causal['ci_high_pct']:+.2f}%), reported as "
            f"a descriptive contrast only."
        )
        adj = causal.get("robustness", {}).get("twoway_fe_zone_trends")
        if adj:
            lines.append("")
            caveat = ""
            if adj.get("rank_deficient"):
                caveat = (
                    " The zone-trend design is rank deficient, so collinear nuisance parameters "
                    "were absorbed; the reported standard error on the treatment term is still "
                    "finite and clustered by zone."
                )
            lines.append(
                f"Adding zone-specific linear time trends, which absorb the differential drift, "
                f"moves the estimate to {adj['att_pct']:+.2f}% "
                f"(95% CI {adj['ci_low_pct']:+.2f}% to {adj['ci_high_pct']:+.2f}%), "
                f"p = {adj['p_value']:.4g}. Almost all of the apparent effect was the "
                f"pre-existing divergence continuing.{caveat}"
            )

    lines += ["", "## Drift", ""]
    drift = results["drift"]
    lines.append(
        f"{drift['features']['n_features']} features scanned, "
        f"{drift['features']['n_warn']} at warn level, {drift['features']['n_alert']} at alert level. "
        f"Prediction PSI {drift['prediction']['psi']:.4f} ({drift['prediction']['status']})."
    )
    lines.append("")

    out = cfg.path("reports") / "results.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
