from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from urbanflow.logging_utils import get_logger, log_event

log = get_logger(__name__)


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < 10 or len(cur) < 10:
        return float("nan")

    edges = np.unique(np.percentile(ref, np.linspace(0, 100, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_pct = np.histogram(cur, bins=edges)[0] / len(cur)
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


@dataclass
class DriftReport:
    table: pd.DataFrame
    n_warn: int
    n_alert: int

    def to_dict(self) -> dict:
        return {
            "n_features": int(len(self.table)),
            "n_warn": self.n_warn,
            "n_alert": self.n_alert,
            "worst": self.table.head(5).to_dict(orient="records"),
        }


def feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_cols: list[str],
    psi_warn: float,
    psi_alert: float,
    ks_alpha: float,
    max_sample: int = 50_000,
) -> DriftReport:
    rng = np.random.default_rng(0)

    def sample(frame: pd.DataFrame, col: str) -> np.ndarray:
        v = frame[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if len(v) > max_sample:
            v = v[rng.choice(len(v), max_sample, replace=False)]
        return v

    rows = []
    for col in feature_cols:
        if col not in reference.columns or col not in current.columns:
            continue
        ref = sample(reference, col)
        cur = sample(current, col)
        if len(ref) < 10 or len(cur) < 10:
            continue
        psi = population_stability_index(ref, cur)
        ks_stat, ks_p = stats.ks_2samp(ref, cur)
        status = "ok"
        if np.isfinite(psi) and psi >= psi_alert:
            status = "alert"
        elif np.isfinite(psi) and psi >= psi_warn:
            status = "warn"
        rows.append(
            {
                "feature": col,
                "psi": round(float(psi), 5),
                "ks_stat": round(float(ks_stat), 5),
                "ks_p": float(ks_p),
                "ks_significant": bool(ks_p < ks_alpha),
                "ref_mean": round(float(ref.mean()), 5),
                "cur_mean": round(float(cur.mean()), 5),
                "status": status,
            }
        )

    table = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    n_warn = int((table["status"] == "warn").sum()) if len(table) else 0
    n_alert = int((table["status"] == "alert").sum()) if len(table) else 0
    log_event(log, "drift scan complete", features=len(table), warn=n_warn, alert=n_alert)
    return DriftReport(table=table, n_warn=n_warn, n_alert=n_alert)


def prediction_drift(
    reference_pred: np.ndarray, current_pred: np.ndarray, psi_alert: float
) -> dict:
    psi = population_stability_index(reference_pred, current_pred)
    ks_stat, ks_p = stats.ks_2samp(reference_pred, current_pred)
    return {
        "psi": round(float(psi), 5),
        "ks_stat": round(float(ks_stat), 5),
        "ks_p": float(ks_p),
        "status": "alert" if np.isfinite(psi) and psi >= psi_alert else "ok",
        "ref_mean": round(float(np.mean(reference_pred)), 4),
        "cur_mean": round(float(np.mean(current_pred)), 4),
    }
