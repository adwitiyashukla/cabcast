from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def bias(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(yhat - y))


def smape(y: np.ndarray, yhat: np.ndarray) -> float:
    denom = (np.abs(y) + np.abs(yhat)) / 2.0
    mask = denom > 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(y[mask] - yhat[mask]) / denom[mask]) * 100.0)


def mase(y: np.ndarray, yhat: np.ndarray, naive: np.ndarray) -> float:
    scale = np.mean(np.abs(y - naive))
    return float(np.mean(np.abs(y - yhat)) / scale) if scale > 0 else float("nan")


def poisson_deviance(y: np.ndarray, yhat: np.ndarray) -> float:
    yhat = np.clip(yhat, 1e-9, None)
    ratio = np.where(y > 0, y / yhat, 1.0)
    term = np.where(y > 0, y * np.log(ratio), 0.0)
    return float(2.0 * np.mean(term - (y - yhat)))


def pinball_loss(y: np.ndarray, yhat: np.ndarray, tau: float) -> float:
    diff = y - yhat
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


def coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y >= lower) & (y <= upper)))


def mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(upper - lower))


def winkler_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> float:
    width = upper - lower
    below = lower - y
    above = y - upper
    penalty = np.where(y < lower, 2.0 / alpha * below, 0.0) + np.where(
        y > upper, 2.0 / alpha * above, 0.0
    )
    return float(np.mean(width + penalty))


def point_metrics(y: np.ndarray, yhat: np.ndarray, naive: np.ndarray | None = None) -> dict:
    out = {
        "mae": mae(y, yhat),
        "rmse": rmse(y, yhat),
        "bias": bias(y, yhat),
        "smape": smape(y, yhat),
        "poisson_deviance": poisson_deviance(y, yhat),
    }
    if naive is not None:
        out["mase"] = mase(y, yhat, naive)
    return out


def interval_metrics(
    y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> dict:
    active = y > 0
    out = {
        "coverage": coverage(y, lower, upper),
        "target_coverage": 1.0 - alpha,
        "mean_width": mean_interval_width(lower, upper),
        "winkler": winkler_score(y, lower, upper, alpha),
    }
    if active.any():
        out["coverage_active"] = coverage(y[active], lower[active], upper[active])
        out["mean_width_active"] = mean_interval_width(lower[active], upper[active])
        out["winkler_active"] = winkler_score(y[active], lower[active], upper[active], alpha)
    return out


def coverage_by_stratum(
    y: np.ndarray, lower: np.ndarray, upper: np.ndarray, strata: np.ndarray
) -> pd.DataFrame:
    df = pd.DataFrame(
        {"y": y, "lower": lower, "upper": upper, "stratum": strata}
    )
    df["covered"] = (df["y"] >= df["lower"]) & (df["y"] <= df["upper"])
    out = df.groupby("stratum", observed=True).agg(
        n=("y", "size"),
        coverage=("covered", "mean"),
        mean_width=("upper", lambda s: float(np.mean(s - df.loc[s.index, "lower"]))),
        mean_actual=("y", "mean"),
    )
    return out.reset_index()
