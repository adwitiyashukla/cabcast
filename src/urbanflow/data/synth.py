from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from urbanflow.data.calendar_ref import holiday_multiplier
from urbanflow.data.profiles import PROFILES
from urbanflow.geo.graph import ZoneGraph
from urbanflow.geo.zones import ZoneSet
from urbanflow.logging_utils import get_logger, log_event

log = get_logger(__name__)

TLC_COLUMNS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "Airport_fee",
]

CRZ_START = dt.date(2025, 1, 5)
CRZ_TRUE_EFFECT = -0.065
CRZ_SPILLOVER_EFFECT = 0.021

GRAVITY_BETA_MINUTES = 11.0
NB_DISPERSION = 6.0
TREND_PER_YEAR = 0.035
YEARLY_AMPLITUDE = 0.11

BASE_FARE = 3.00
PER_MILE = 3.50
PER_SLOW_MINUTE = 0.70
MTA_TAX = 0.50
IMPROVEMENT_SURCHARGE = 1.00
CONGESTION_SURCHARGE = 2.50
AIRPORT_FEE = 1.75

DIRT_DUPLICATE_RATE = 0.0022
DIRT_NULL_PASSENGER_RATE = 0.0140
DIRT_ZERO_DISTANCE_RATE = 0.0075
DIRT_NEGATIVE_FARE_RATE = 0.0016
DIRT_ABSURD_DURATION_RATE = 0.0009
DIRT_OUT_OF_MONTH_RATE = 0.0006


def month_range(start_month: str, end_month: str) -> list[tuple[int, int]]:
    y0, m0 = (int(x) for x in start_month.split("-"))
    y1, m1 = (int(x) for x in end_month.split("-"))
    out: list[tuple[int, int]] = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        out.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def month_bounds(year: int, month: int) -> tuple[dt.datetime, dt.datetime]:
    start = dt.datetime(year, month, 1)
    end = dt.datetime(year + (month == 12), 1 if month == 12 else month + 1, 1)
    return start, end


def generate_weather(start: dt.datetime, end: dt.datetime, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="h", inclusive="left")
    n = len(idx)
    doy = idx.dayofyear.to_numpy()
    hour = idx.hour.to_numpy()

    seasonal = 12.8 - 13.2 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    diurnal = 4.4 * np.sin(2 * np.pi * (hour - 9) / 24.0)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.92 * noise[i - 1] + rng.normal(0, 1.15)
    temp = np.clip(seasonal + diurnal + noise, -18.0, 40.0)

    storm = np.zeros(n)
    n_storms = max(int(n / 168 * 2.4), 1)
    for _ in range(n_storms):
        s = rng.integers(0, n)
        dur = int(rng.integers(2, 14))
        peak = float(rng.gamma(2.0, 1.1))
        window = np.arange(dur)
        shape = np.exp(-((window - dur / 3) ** 2) / (2 * (dur / 3.2) ** 2))
        storm[s : s + dur] += peak * shape[: max(0, min(dur, n - s))]
    precip = np.clip(storm, 0, 12.0)
    snow = np.where(temp < 1.0, np.clip(precip * 7.2, 0, 9.0), 0.0)
    rain = np.where(temp >= 1.0, precip, 0.0)

    wind = np.clip(rng.gamma(2.4, 3.4, n) + 2.0 * (precip > 0.4), 0, None)
    humidity = np.clip(58 + 22 * (precip > 0.1) + rng.normal(0, 9, n), 8, 100)
    cloud = np.clip(38 + 46 * (precip > 0.05) + rng.normal(0, 20, n), 0, 100)

    return pd.DataFrame(
        {
            "hour_ts": idx,
            "temperature_2m": np.round(temp, 2),
            "precipitation": np.round(rain, 3),
            "snowfall": np.round(snow, 3),
            "wind_speed_10m": np.round(wind, 2),
            "relative_humidity_2m": np.round(humidity, 1),
            "cloud_cover": np.round(cloud, 1),
        }
    )


def _zone_kind(zones: ZoneSet) -> np.ndarray:
    kinds = np.where(
        zones.gdf["is_airport"].to_numpy(),
        "airport",
        np.where(zones.gdf["borough"].to_numpy() == "Manhattan", "core", "outer"),
    )
    return kinds


def _weather_multiplier(w: pd.DataFrame) -> np.ndarray:
    rain = w["precipitation"].to_numpy()
    snow = w["snowfall"].to_numpy()
    temp = w["temperature_2m"].to_numpy()
    m = 1.0 + 0.085 * np.tanh(rain / 1.2) - 0.16 * np.tanh(snow / 3.0)
    m *= 1.0 + 0.030 * np.tanh((5.0 - temp) / 12.0)
    m *= 1.0 - 0.025 * np.tanh((temp - 29.0) / 6.0)
    return m


def _intensity_matrix(
    zones: ZoneSet,
    hours: pd.DatetimeIndex,
    weather: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    kinds = _zone_kind(zones)
    profile_stack = np.stack([PROFILES[k] for k in kinds])
    how = (hours.dayofweek.to_numpy() * 24 + hours.hour.to_numpy()).astype(int)
    temporal = profile_stack[:, how]

    weight = zones.gdf["demand_weight"].to_numpy()[:, None]

    days = hours.normalize()
    origin = pd.Timestamp("2024-01-01")
    years_elapsed = (days - origin).days.to_numpy() / 365.25
    trend = (1.0 + TREND_PER_YEAR) ** years_elapsed
    yearly = 1.0 + YEARLY_AMPLITUDE * np.sin(
        2 * np.pi * (hours.dayofyear.to_numpy() - 105) / 365.25
    )
    hol = np.array([holiday_multiplier(d.date()) for d in days])

    wx = weather.set_index("hour_ts").reindex(hours)
    wx = wx.ffill().bfill()
    weather_mult = _weather_multiplier(wx)

    in_crz = zones.gdf["in_crz"].to_numpy()
    manhattan = (zones.gdf["borough"].to_numpy() == "Manhattan") & (~in_crz)
    post = (days.date >= CRZ_START).astype(float)
    crz = np.ones((len(zones.gdf), len(hours)))
    crz[in_crz] = 1.0 + CRZ_TRUE_EFFECT * post
    crz[manhattan] = 1.0 + CRZ_SPILLOVER_EFFECT * post

    zone_noise = rng.lognormal(0.0, 0.20, size=(len(zones.gdf), 1))

    lam = weight * temporal * trend * yearly * hol * weather_mult * crz * zone_noise
    return np.maximum(lam, 1e-6)


def _sample_destinations(
    origin_idx: np.ndarray,
    graph: ZoneGraph,
    attract: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    utility = np.log(attract[None, :] + 1e-9) - graph.travel_minutes / GRAVITY_BETA_MINUTES
    probs = np.exp(utility - utility.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    cdf = np.cumsum(probs, axis=1)

    out = np.empty(len(origin_idx), dtype=np.int32)
    order = np.argsort(origin_idx, kind="stable")
    sorted_origins = origin_idx[order]
    uniques, starts = np.unique(sorted_origins, return_index=True)
    bounds = np.append(starts, len(sorted_origins))
    for k, o in enumerate(uniques):
        n = bounds[k + 1] - bounds[k]
        u = rng.random(n)
        picks = np.searchsorted(cdf[o], u).clip(0, cdf.shape[1] - 1)
        out[order[bounds[k] : bounds[k + 1]]] = picks
    return out


def generate_month(
    year: int,
    month: int,
    zones: ZoneSet,
    graph: ZoneGraph,
    weather: pd.DataFrame,
    trips_target: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + year * 100 + month)
    start, end = month_bounds(year, month)
    hours = pd.date_range(start, end, freq="h", inclusive="left")

    lam = _intensity_matrix(zones, hours, weather, rng)
    lam *= trips_target / lam.sum()

    r = NB_DISPERSION
    p = r / (r + lam)
    counts = rng.negative_binomial(r, p).astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        return pd.DataFrame(columns=TLC_COLUMNS)

    zone_pos, hour_pos = np.nonzero(counts)
    reps = counts[zone_pos, hour_pos]
    origin_idx = np.repeat(zone_pos, reps).astype(np.int32)
    hour_idx = np.repeat(hour_pos, reps).astype(np.int32)

    offsets = rng.random(total) * 3600.0
    pickup = hours.values[hour_idx] + (offsets * 1e9).astype("timedelta64[ns]")

    attract = zones.gdf["demand_weight"].to_numpy()
    dest_idx = _sample_destinations(origin_idx, graph, attract, rng)

    base_minutes = graph.travel_minutes[origin_idx, dest_idx]
    hour_of_day = hours.hour.to_numpy()[hour_idx]
    congestion = 1.0 + 0.42 * np.exp(-((hour_of_day - 8.5) ** 2) / 8.0)
    congestion += 0.52 * np.exp(-((hour_of_day - 17.8) ** 2) / 9.0)
    duration_min = np.maximum(
        base_minutes * congestion * rng.lognormal(0.0, 0.30, total), 1.0
    )

    straight_km = base_minutes / 60.0 * 24.0
    detour = rng.lognormal(0.11, 0.16, total)
    miles = np.maximum(straight_km * 0.621371 * detour, 0.15)

    free_minutes = miles / 24.0 * 60.0 / 0.621371
    slow_minutes = np.maximum(duration_min - free_minutes, 0.0)
    fare = BASE_FARE + PER_MILE * miles + PER_SLOW_MINUTE * slow_minutes
    fare *= rng.lognormal(0.0, 0.06, total)
    fare = np.round(fare, 2)

    night = (hour_of_day >= 20) | (hour_of_day < 6)
    rush = (~night) & (np.isin(hour_of_day, np.arange(16, 20)))
    dow = hours.dayofweek.to_numpy()[hour_idx]
    extra = np.where(night, 1.00, 0.0) + np.where(rush & (dow < 5), 2.50, 0.0)

    in_crz = zones.gdf["in_crz"].to_numpy()
    is_airport = zones.gdf["is_airport"].to_numpy()
    touches_crz = in_crz[origin_idx] | in_crz[dest_idx]
    congestion_surcharge = np.where(touches_crz, CONGESTION_SURCHARGE, 0.0)
    airport_fee = np.where(is_airport[origin_idx], AIRPORT_FEE, 0.0)

    tolls = np.where(rng.random(total) < 0.055, np.round(rng.gamma(3.0, 2.6, total), 2), 0.0)
    payment_type = np.where(rng.random(total) < 0.72, 1, 2)
    tip_rate = np.clip(rng.normal(0.212, 0.075, total), 0.0, 0.6)
    tip = np.where(payment_type == 1, np.round(fare * tip_rate, 2), 0.0)

    mta = np.full(total, MTA_TAX)
    improvement = np.full(total, IMPROVEMENT_SURCHARGE)
    total_amount = np.round(
        fare + extra + mta + tip + tolls + improvement + congestion_surcharge + airport_fee, 2
    )

    passengers = rng.choice(
        [1, 2, 3, 4, 5, 6], size=total, p=[0.715, 0.148, 0.042, 0.021, 0.048, 0.026]
    )
    ratecode = np.where(
        is_airport[origin_idx] | is_airport[dest_idx],
        rng.choice([1, 2], size=total, p=[0.35, 0.65]),
        1,
    )
    zone_ids = zones.zone_ids

    df = pd.DataFrame(
        {
            "VendorID": rng.choice([1, 2], size=total, p=[0.31, 0.69]).astype("int32"),
            "tpep_pickup_datetime": pickup,
            "tpep_dropoff_datetime": pickup
            + (duration_min * 60.0 * 1e9).astype("timedelta64[ns]"),
            "passenger_count": passengers.astype("float64"),
            "trip_distance": np.round(miles, 2),
            "RatecodeID": ratecode.astype("float64"),
            "store_and_fwd_flag": np.where(rng.random(total) < 0.006, "Y", "N"),
            "PULocationID": zone_ids[origin_idx].astype("int32"),
            "DOLocationID": zone_ids[dest_idx].astype("int32"),
            "payment_type": payment_type.astype("int64"),
            "fare_amount": fare,
            "extra": extra,
            "mta_tax": mta,
            "tip_amount": tip,
            "tolls_amount": tolls,
            "improvement_surcharge": improvement,
            "total_amount": total_amount,
            "congestion_surcharge": congestion_surcharge,
            "Airport_fee": airport_fee,
        }
    )
    return _inject_quality_issues(df, rng, start, end)


def _inject_quality_issues(
    df: pd.DataFrame, rng: np.random.Generator, start: dt.datetime, end: dt.datetime
) -> pd.DataFrame:
    n = len(df)

    idx = rng.choice(n, size=int(n * DIRT_NULL_PASSENGER_RATE), replace=False)
    df.loc[df.index[idx], "passenger_count"] = np.nan

    idx = rng.choice(n, size=int(n * DIRT_ZERO_DISTANCE_RATE), replace=False)
    df.loc[df.index[idx], "trip_distance"] = 0.0

    idx = rng.choice(n, size=int(n * DIRT_NEGATIVE_FARE_RATE), replace=False)
    df.loc[df.index[idx], ["fare_amount", "total_amount"]] *= -1.0

    idx = rng.choice(n, size=int(n * DIRT_ABSURD_DURATION_RATE), replace=False)
    df.loc[df.index[idx], "tpep_dropoff_datetime"] = df.loc[
        df.index[idx], "tpep_pickup_datetime"
    ] + pd.to_timedelta(rng.integers(30, 96, len(idx)), unit="h")

    idx = rng.choice(n, size=int(n * DIRT_OUT_OF_MONTH_RATE), replace=False)
    shift = pd.to_timedelta(rng.choice([-1, 1], len(idx)) * rng.integers(40, 900, len(idx)), unit="D")
    df.loc[df.index[idx], "tpep_pickup_datetime"] = (
        df.loc[df.index[idx], "tpep_pickup_datetime"] + shift
    )

    n_dup = int(n * DIRT_DUPLICATE_RATE)
    if n_dup:
        dups = df.iloc[rng.choice(n, size=n_dup, replace=False)]
        df = pd.concat([df, dups], ignore_index=True)

    return df.sample(frac=1.0, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)


def write_synthetic_bronze(
    cfg,
    zones: ZoneSet,
    graph: ZoneGraph,
    months: list[tuple[int, int]] | None = None,
    trips_per_month: int | None = None,
) -> dict[str, int]:
    bronze = cfg.path("bronze")
    external = cfg.path("external")
    months = months or month_range(cfg.data.start_month, cfg.data.end_month)
    target = int(trips_per_month or cfg.data.synthetic_trips_per_month)
    seed = int(cfg.project.random_seed)

    w_start = month_bounds(*months[0])[0] - dt.timedelta(days=31)
    w_end = month_bounds(*months[-1])[1] + dt.timedelta(days=1)
    weather = generate_weather(w_start, w_end, seed)
    weather.to_parquet(external / "weather_hourly.parquet", index=False)

    written: dict[str, int] = {}
    for year, month in months:
        tag = f"{year}-{month:02d}"
        out = bronze / f"{cfg.data.service}_tripdata_{tag}.parquet"
        df = generate_month(year, month, zones, graph, weather, target, seed)
        df.to_parquet(out, index=False, compression="zstd")
        written[tag] = len(df)
        log_event(log, "synthetic month written", month=tag, rows=len(df), mb=round(out.stat().st_size / 1e6, 1))

    log_event(log, "synthetic bronze complete", months=len(written), rows=sum(written.values()))
    return written


def synthetic_manifest(written: dict[str, int], path: Path) -> None:
    pd.DataFrame(
        {"month": list(written), "rows": list(written.values()), "source": "synthetic"}
    ).to_csv(path, index=False)
