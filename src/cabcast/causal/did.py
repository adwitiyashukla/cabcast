from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from cabcast.logging_utils import get_logger, log_event

log = get_logger(__name__)


@dataclass
class DiDResult:
    att_log: float
    att_pct: float
    std_error: float
    t_stat: float
    p_value: float
    ci_low_pct: float
    ci_high_pct: float
    n_obs: int
    n_zones: int
    n_periods: int
    spec: str = "twoway_fixed_effects"
    rank_deficient: bool = False
    standard_error_is_finite: bool = True
    status: str = "ok"
    pretrend_p_value: float = float("nan")
    pretrend_slope_pct_per_period: float = float("nan")
    pretrend_significant_share: float = float("nan")
    parallel_trends_holds: bool = True
    robustness: dict = field(default_factory=dict)
    event_study: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> dict:
        return {
            "att_log": round(self.att_log, 6),
            "att_pct": round(self.att_pct, 4),
            "std_error": round(self.std_error, 6),
            "t_stat": round(self.t_stat, 4),
            "p_value": float(self.p_value),
            "ci_low_pct": round(self.ci_low_pct, 4),
            "ci_high_pct": round(self.ci_high_pct, 4),
            "n_obs": self.n_obs,
            "n_zones": self.n_zones,
            "n_periods": self.n_periods,
            "spec": self.spec,
            "rank_deficient": bool(self.rank_deficient),
            "standard_error_is_finite": bool(self.standard_error_is_finite),
            "status": self.status,
            "pretrend_p_value": float(self.pretrend_p_value),
            "pretrend_slope_pct_per_period": float(self.pretrend_slope_pct_per_period),
            "pretrend_significant_share": float(self.pretrend_significant_share),
            "parallel_trends_holds": bool(self.parallel_trends_holds),
        }


def build_did_panel(
    panel: pd.DataFrame,
    treatment_start: str,
    window_months: int,
    control_boroughs: list[str] | None = None,
    min_weekly_trips: float = 0.0,
    freq: str = "W",
) -> pd.DataFrame:
    start = pd.Timestamp(treatment_start)
    lo = start - pd.DateOffset(months=window_months)
    hi = start + pd.DateOffset(months=window_months)

    df = panel[(panel["hour_ts"] >= lo) & (panel["hour_ts"] < hi)].copy()
    df["period"] = df["hour_ts"].dt.to_period(freq).dt.start_time

    agg = (
        df.groupby(["zone_id", "period"], observed=True)
        .agg(
            trips=("trips", "sum"),
            in_crz=("in_crz", "first"),
            borough=("borough", "first"),
        )
        .reset_index()
    )
    controls = list(control_boroughs or ["Manhattan"])
    keep = agg["in_crz"] | agg["borough"].isin(controls)
    agg = agg[keep].copy()

    volume = agg.groupby("zone_id", observed=True)["trips"].transform("median")
    agg = agg[volume >= float(min_weekly_trips)].copy()

    complete = agg.groupby("zone_id", observed=True)["period"].transform("size")
    agg = agg[complete == agg["period"].nunique()].copy()

    agg["treated"] = agg["in_crz"].astype(int)
    agg["post"] = (agg["period"] >= start).astype(int)
    agg["treat_post"] = agg["treated"] * agg["post"]
    agg["log_trips"] = np.log(agg["trips"] + 1.0)

    periods = np.sort(np.asarray(agg["period"].unique(), dtype="datetime64[ns]"))
    ref_idx = int(np.searchsorted(periods, np.datetime64(start, "ns")))
    lookup = {pd.Timestamp(p): i - ref_idx for i, p in enumerate(periods)}
    agg["event_time"] = agg["period"].map(lookup)
    return agg


MIN_ZONES_PER_ARM = 3
MIN_PERIODS = 6
PRETREND_ALPHA = 0.05
PRETREND_MAX_SIG_SHARE = 0.25


def _degenerate(agg: pd.DataFrame, reason: str) -> DiDResult:
    log_event(log, "DiD skipped", reason=reason, rows=len(agg))
    nan = float("nan")
    return DiDResult(
        att_log=nan, att_pct=nan, std_error=nan, t_stat=nan, p_value=nan,
        ci_low_pct=nan, ci_high_pct=nan, n_obs=int(len(agg)),
        n_zones=int(agg["zone_id"].nunique()) if len(agg) else 0,
        n_periods=int(agg["period"].nunique()) if len(agg) else 0,
        status=reason,
        event_study=pd.DataFrame(
            columns=["event_time", "coef_log", "coef_pct", "ci_low_pct", "ci_high_pct", "p_value"]
        ),
    )


def panel_is_estimable(agg: pd.DataFrame) -> str | None:
    if len(agg) == 0:
        return "empty_panel"
    treated = agg.loc[agg["treated"] == 1, "zone_id"].nunique()
    control = agg.loc[agg["treated"] == 0, "zone_id"].nunique()
    if treated < MIN_ZONES_PER_ARM or control < MIN_ZONES_PER_ARM:
        return "too_few_zones_per_arm"
    if agg["period"].nunique() < MIN_PERIODS:
        return "too_few_periods"
    if agg["post"].nunique() < 2:
        return "no_post_period"
    return None


def estimate_did(
    agg: pd.DataFrame, cluster_by: str = "zone_id", zone_trends: bool = False
) -> DiDResult:
    formula = "log_trips ~ treat_post + C(zone_id) + C(period)"
    if zone_trends:
        formula += " + C(zone_id):event_time"
    model = smf.ols(formula, data=agg)
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": agg[cluster_by]})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        coef = float(fit.params["treat_post"])
        se = float(fit.bse["treat_post"])
        ci = fit.conf_int().loc["treat_post"]

    n_params = int(fit.params.shape[0])
    rank = int(getattr(fit.model, "rank", n_params))
    deficient = rank < n_params
    se_finite = bool(np.isfinite(se) and se > 0)
    if deficient:
        log_event(
            log,
            "design matrix is rank deficient, collinear nuisance parameters were absorbed",
            spec="twoway_fe_zone_trends" if zone_trends else "twoway_fixed_effects",
            params=n_params,
            rank=rank,
            treat_post_se_finite=se_finite,
        )

    result = DiDResult(
        att_log=coef,
        att_pct=(np.exp(coef) - 1.0) * 100.0,
        std_error=se,
        t_stat=float(fit.tvalues["treat_post"]),
        p_value=float(fit.pvalues["treat_post"]),
        ci_low_pct=(np.exp(float(ci[0])) - 1.0) * 100.0,
        ci_high_pct=(np.exp(float(ci[1])) - 1.0) * 100.0,
        n_obs=int(fit.nobs),
        n_zones=int(agg["zone_id"].nunique()),
        n_periods=int(agg["period"].nunique()),
        spec="twoway_fe_zone_trends" if zone_trends else "twoway_fixed_effects",
        rank_deficient=deficient,
        standard_error_is_finite=se_finite,
    )
    log_event(log, "DiD estimated", **result.to_dict())
    return result


def event_study(agg: pd.DataFrame, cluster_by: str = "zone_id") -> tuple[pd.DataFrame, float]:
    df = agg.copy()
    df["evt"] = df["event_time"].astype(int)
    reference = -1
    levels = sorted(t for t in df["evt"].unique() if t != reference)

    for t in levels:
        df[f"d_{t}".replace("-", "m")] = ((df["evt"] == t) & (df["treated"] == 1)).astype(int)
    dummies = [f"d_{t}".replace("-", "m") for t in levels]

    formula = "log_trips ~ " + " + ".join(dummies) + " + C(zone_id) + C(period)"
    fit = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df[cluster_by]}
    )

    rows = []
    for t, name in zip(levels, dummies, strict=True):
        ci = fit.conf_int().loc[name]
        rows.append(
            {
                "event_time": t,
                "coef_log": float(fit.params[name]),
                "coef_pct": (np.exp(float(fit.params[name])) - 1.0) * 100.0,
                "ci_low_pct": (np.exp(float(ci[0])) - 1.0) * 100.0,
                "ci_high_pct": (np.exp(float(ci[1])) - 1.0) * 100.0,
                "p_value": float(fit.pvalues[name]),
            }
        )
    rows.append(
        {
            "event_time": reference,
            "coef_log": 0.0,
            "coef_pct": 0.0,
            "ci_low_pct": 0.0,
            "ci_high_pct": 0.0,
            "p_value": float("nan"),
        }
    )
    table = pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)

    pre = [n for t, n in zip(levels, dummies, strict=True) if t < reference]
    pretrend_p = float("nan")
    if pre:
        test = fit.f_test(" = ".join(pre) + " = 0")
        pretrend_p = float(np.ravel(test.pvalue)[0])
    log_event(log, "event study complete", leads_lags=len(table), pretrend_p=pretrend_p)
    return table, pretrend_p


def run_congestion_pricing_study(panel: pd.DataFrame, cfg) -> DiDResult:
    agg = build_did_panel(
        panel,
        str(cfg.causal.treatment_start),
        int(cfg.causal.event_window_months),
        control_boroughs=list(cfg.causal.get("control_boroughs", ["Manhattan"])),
        min_weekly_trips=float(cfg.causal.get("min_weekly_trips", 0.0)),
    )
    reason = panel_is_estimable(agg)
    if reason:
        return _degenerate(agg, reason)

    result = estimate_did(agg)
    table, pretrend_p = event_study(agg)
    result.event_study = table
    result.pretrend_p_value = pretrend_p

    pre = table[table["event_time"] < -1]
    result.pretrend_slope_pct_per_period = (
        float(np.polyfit(pre["event_time"], pre["coef_pct"], 1)[0]) if len(pre) > 3 else float("nan")
    )
    result.pretrend_significant_share = (
        float((pre["p_value"] < 0.05).mean()) if len(pre) else float("nan")
    )
    result.parallel_trends_holds = bool(
        pretrend_p > PRETREND_ALPHA and result.pretrend_significant_share < PRETREND_MAX_SIG_SHARE
    )

    if not result.parallel_trends_holds:
        log_event(
            log,
            "PARALLEL TRENDS VIOLATED, naive DiD is not identified",
            pretrend_p=pretrend_p,
            pretrend_slope_pct_per_period=round(result.pretrend_slope_pct_per_period, 4),
            share_of_pre_periods_significant=round(result.pretrend_significant_share, 3),
        )
        adjusted = estimate_did(agg, zone_trends=True)
        payload = {
            k: v
            for k, v in adjusted.to_dict().items()
            if k
            not in {
                "event_study",
                "parallel_trends_holds",
                "pretrend_p_value",
                "pretrend_slope_pct_per_period",
                "pretrend_significant_share",
            }
        }
        result.robustness = {"twoway_fe_zone_trends": payload}
        log_event(log, "zone-trend adjusted estimate", **result.robustness["twoway_fe_zone_trends"])
    return result
