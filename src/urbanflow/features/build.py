from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import scipy.sparse as sp

from urbanflow.data.calendar_ref import holiday_name
from urbanflow.geo.graph import ZoneGraph
from urbanflow.logging_utils import get_logger, log_event, stage

log = get_logger(__name__)

TARGET = "trips"
KEY = ["zone_id", "hour_ts"]

PASSTHROUGH = [
    "temperature_2m",
    "precipitation",
    "snowfall",
    "wind_speed_10m",
    "relative_humidity_2m",
    "cloud_cover",
    "area_sqkm",
    "km_from_midtown",
    "centroid_lon",
    "centroid_lat",
    "graph_degree",
    "graph_betweenness",
    "graph_closeness",
    "graph_pagerank",
    "graph_eigencentrality",
]

LEAKY = [
    "arrivals",
    "mean_fare",
    "mean_miles",
    "mean_duration_min",
    "mean_tip_rate",
    "mean_congestion_surcharge",
    "mean_passengers",
]


def _calendar(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["hour_ts"]
    df["hour"] = ts.dt.hour.astype("int16")
    df["dow"] = ts.dt.dayofweek.astype("int16")
    df["month"] = ts.dt.month.astype("int16")
    df["day_of_year"] = ts.dt.dayofyear.astype("int16")
    df["week_of_year"] = ts.dt.isocalendar().week.astype("int16")
    df["hour_of_week"] = (df["dow"] * 24 + df["hour"]).astype("int16")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype("int8")
    df["is_am_peak"] = df["hour"].between(7, 9).astype("int8")
    df["is_pm_peak"] = df["hour"].between(16, 19).astype("int8")

    days = ts.dt.date
    names = {d: holiday_name(d) for d in days.unique()}
    df["is_holiday"] = days.map(lambda d: names[d] is not None).astype("int8")
    unique_days = pd.Index(sorted(names))
    hol_days = pd.Index([d for d in unique_days if names[d] is not None])
    if len(hol_days):
        pos = np.searchsorted(hol_days, unique_days)
        prev_gap = np.array(
            [
                (unique_days[i] - hol_days[max(pos[i] - 1, 0)]).days if pos[i] > 0 else 99
                for i in range(len(unique_days))
            ]
        )
        next_gap = np.array(
            [
                (hol_days[pos[i]] - unique_days[i]).days if pos[i] < len(hol_days) else 99
                for i in range(len(unique_days))
            ]
        )
        gap = pd.Series(np.minimum(prev_gap, next_gap).clip(0, 14), index=unique_days)
        df["days_to_holiday"] = days.map(gap).astype("int16")
    else:
        df["days_to_holiday"] = np.int16(99)
    return df


def _fourier(df: pd.DataFrame, cfg) -> pd.DataFrame:
    terms = cfg.features.fourier_terms
    for k in range(1, int(terms.daily) + 1):
        ang = 2 * np.pi * k * df["hour"] / 24.0
        df[f"fourier_day_sin_{k}"] = np.sin(ang)
        df[f"fourier_day_cos_{k}"] = np.cos(ang)
    for k in range(1, int(terms.weekly) + 1):
        ang = 2 * np.pi * k * df["hour_of_week"] / 168.0
        df[f"fourier_week_sin_{k}"] = np.sin(ang)
        df[f"fourier_week_cos_{k}"] = np.cos(ang)
    for k in range(1, int(terms.yearly) + 1):
        ang = 2 * np.pi * k * df["day_of_year"] / 365.25
        df[f"fourier_year_sin_{k}"] = np.sin(ang)
        df[f"fourier_year_cos_{k}"] = np.cos(ang)
    return df


def _lags_and_rolling(df: pd.DataFrame, cfg) -> pd.DataFrame:
    g = df.groupby("zone_id", sort=False)[TARGET]
    for lag in cfg.features.lags_hours:
        df[f"lag_{lag}h"] = g.shift(lag).astype("float32")

    base = df.groupby("zone_id", sort=False)[TARGET].shift(1)
    holder = pd.DataFrame({"zone_id": df["zone_id"].to_numpy(), "v": base.to_numpy()})
    grp = holder.groupby("zone_id", sort=False)["v"]
    for window in cfg.features.rolling_windows_hours:
        roll = grp.rolling(window, min_periods=max(2, window // 4))
        df[f"roll_mean_{window}h"] = roll.mean().to_numpy().astype("float32")
        df[f"roll_std_{window}h"] = roll.std().to_numpy().astype("float32")
        df[f"roll_max_{window}h"] = roll.max().to_numpy().astype("float32")

    df["lag_ratio_1_24"] = df["lag_1h"] / (df["lag_24h"] + 1.0)
    df["lag_ratio_24_168"] = df["lag_24h"] / (df["lag_168h"] + 1.0)
    df["lag_diff_24_48"] = df["lag_24h"] - df["lag_48h"]
    df["dev_from_roll_168"] = df["lag_1h"] - df["roll_mean_168h"]

    for col in LEAKY:
        if col in df.columns:
            df[f"{col}_lag24"] = (
                df.groupby("zone_id", sort=False)[col].shift(24).astype("float32")
            )
    return df


def _spatial_diffusion(df: pd.DataFrame, graph: ZoneGraph, cfg) -> pd.DataFrame:
    zone_ids = np.sort(df["zone_id"].unique())
    idx = np.searchsorted(graph.zone_ids, zone_ids)
    sub = graph.adjacency[idx][:, idx].astype(np.float64)
    deg = np.asarray(sub.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    p = sp.diags(1.0 / deg) @ sub

    wide = df.pivot_table(index="hour_ts", columns="zone_id", values="lag_1h", observed=True)
    wide = wide.reindex(columns=zone_ids).fillna(0.0)
    mat = wide.to_numpy()

    current = mat
    frames = {}
    for step in cfg.features.spatial_diffusion_steps:
        for _ in range(step if not frames else 1):
            current = current @ p.T
        frames[step] = pd.DataFrame(current, index=wide.index, columns=zone_ids)

    for step, frame in frames.items():
        melted = frame.stack().rename(f"spatial_lag1_hop{step}").reset_index()
        melted.columns = ["hour_ts", "zone_id", f"spatial_lag1_hop{step}"]
        df = df.merge(melted, on=["hour_ts", "zone_id"], how="left")

    first = f"spatial_lag1_hop{cfg.features.spatial_diffusion_steps[0]}"
    df["spatial_excess"] = df["lag_1h"] - df[first]
    return df


def _weather_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df["is_raining"] = (df["precipitation"] > 0.1).astype("int8")
    df["is_snowing"] = (df["snowfall"] > 0.1).astype("int8")
    df["is_freezing"] = (df["temperature_2m"] < 0.0).astype("int8")
    df["rain_x_peak"] = df["is_raining"] * (df["is_am_peak"] | df["is_pm_peak"])
    df["temp_sq"] = df["temperature_2m"] ** 2
    df["rain_x_crz"] = df["is_raining"] * df["in_crz"].astype("int8")
    return df


def build_features(panel: pd.DataFrame, graph: ZoneGraph, cfg) -> tuple[pd.DataFrame, list[str]]:
    with stage(log, "build_features", rows_in=len(panel)) as st:
        df = panel.sort_values(KEY).reset_index(drop=True).copy()
        df["in_crz"] = df["in_crz"].astype(bool)
        df["is_airport_flag"] = df["is_airport"].astype("int8")
        df["in_crz_flag"] = df["in_crz"].astype("int8")

        df = _calendar(df)
        df = _fourier(df, cfg)
        df = _lags_and_rolling(df, cfg)
        df = _spatial_diffusion(df, graph, cfg)
        df = _weather_interactions(df)

        warmup = int(max(cfg.features.lags_hours))
        cutoff = df["hour_ts"].min() + pd.Timedelta(hours=warmup)
        before = len(df)
        df = df[df["hour_ts"] >= cutoff].reset_index(drop=True)

        eig_cols = [c for c in df.columns if c.startswith("lap_eig_")]
        lag_cols = [c for c in df.columns if c.startswith(("lag_", "roll_", "spatial_", "dev_"))]
        cal_cols = [
            "hour", "dow", "month", "day_of_year", "week_of_year", "hour_of_week",
            "is_weekend", "is_night", "is_am_peak", "is_pm_peak", "is_holiday",
            "days_to_holiday",
        ]
        fourier_cols = [c for c in df.columns if c.startswith("fourier_")]
        weather_cols = [
            "is_raining", "is_snowing", "is_freezing", "rain_x_peak", "temp_sq", "rain_x_crz",
        ]
        leaky_lagged = [c for c in df.columns if c.endswith("_lag24")]
        zone_cols = ["is_airport_flag", "in_crz_flag"]

        feature_cols = (
            cal_cols + fourier_cols + lag_cols + eig_cols + PASSTHROUGH
            + weather_cols + leaky_lagged + zone_cols
        )
        feature_cols = [c for c in dict.fromkeys(feature_cols) if c in df.columns]

        for col in feature_cols:
            df[col] = df[col].astype("float32")

        keep = ["zone_id", "hour_ts", TARGET, "hour_of_week", *feature_cols]
        df = df[[c for c in dict.fromkeys(keep) if c in df.columns]]
        df = df.sort_values(["hour_ts", "zone_id"], kind="stable").reset_index(drop=True)
        gc.collect()

        st["rows_out"] = len(df)
        st["dropped_warmup"] = before - len(df)
        st["n_features"] = len(feature_cols)

    log_event(log, "feature matrix ready", rows=len(df), features=len(feature_cols))
    return df, feature_cols


def assert_no_leakage(feature_cols: list[str]) -> None:
    banned = set(LEAKY) | {TARGET}
    offenders = [c for c in feature_cols if c in banned]
    if offenders:
        raise ValueError(f"contemporaneous columns leaked into features: {offenders}")
