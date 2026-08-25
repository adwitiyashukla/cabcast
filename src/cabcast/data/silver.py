from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import pandas as pd

from cabcast.logging_utils import get_logger, log_event, stage
from cabcast.quality.contracts import QualityReport

log = get_logger(__name__)

def _duckdb_threads(cfg) -> int:
    configured = int(cfg.project.get("num_threads", 0))
    return configured if configured > 0 else max(os.cpu_count() or 4, 1)


REASON_SQL = {
    "null_timestamps": "pickup_ts IS NULL OR dropoff_ts IS NULL",
    "missing_zone": "pickup_zone IS NULL OR dropoff_zone IS NULL",
    "missing_distance": "trip_miles IS NULL",
    "missing_fare": "fare_amount IS NULL OR total_amount IS NULL",
    "nonpositive_duration": "trip_seconds IS NOT NULL AND trip_seconds <= 0",
    "duration_too_short": "trip_seconds IS NOT NULL AND trip_seconds < $min_sec",
    "duration_too_long": "trip_seconds IS NOT NULL AND trip_seconds > $max_sec",
    "distance_too_short": "trip_miles IS NOT NULL AND trip_miles < $min_mi",
    "distance_too_long": "trip_miles IS NOT NULL AND trip_miles > $max_mi",
    "fare_out_of_range": "fare_amount IS NOT NULL AND (fare_amount < $min_fare OR fare_amount > $max_fare)",
    "negative_total": "total_amount IS NOT NULL AND total_amount < 0",
    "negative_tip": "COALESCE(tip_amount, 0) < 0",
    "bad_zone": "pickup_zone IS NOT NULL AND dropoff_zone IS NOT NULL AND (pickup_zone NOT BETWEEN 1 AND 265 OR dropoff_zone NOT BETWEEN 1 AND 265)",
    "too_many_passengers": "COALESCE(passenger_count, 1) > $max_pax",
    "outside_month_window": "pickup_ts IS NOT NULL AND (pickup_ts < $win_start OR pickup_ts >= $win_end)",
}


def substitute(text: str, params: dict) -> str:
    for key, value in params.items():
        text = text.replace(f"${key}", f"'{value}'" if isinstance(value, str) else str(value))
    return text


def reject_clause(params: dict) -> str:
    return " OR ".join(f"COALESCE({substitute(v, params)}, FALSE)" for v in REASON_SQL.values())


def _select_expr() -> str:
    return """
        CAST(tpep_pickup_datetime AS TIMESTAMP)              AS pickup_ts,
        CAST(tpep_dropoff_datetime AS TIMESTAMP)             AS dropoff_ts,
        CAST(PULocationID AS INTEGER)                        AS pickup_zone,
        CAST(DOLocationID AS INTEGER)                        AS dropoff_zone,
        CAST(passenger_count AS DOUBLE)                      AS passenger_count,
        CAST(trip_distance AS DOUBLE)                        AS trip_miles,
        CAST(fare_amount AS DOUBLE)                          AS fare_amount,
        CAST(tip_amount AS DOUBLE)                           AS tip_amount,
        CAST(total_amount AS DOUBLE)                         AS total_amount,
        CAST(COALESCE(congestion_surcharge, 0) AS DOUBLE)    AS congestion_surcharge,
        CAST(payment_type AS INTEGER)                        AS payment_type,
        date_diff('second',
                  CAST(tpep_pickup_datetime AS TIMESTAMP),
                  CAST(tpep_dropoff_datetime AS TIMESTAMP)) AS trip_seconds
    """


def build_silver(cfg, months: list[str] | None = None) -> tuple[Path, QualityReport]:
    bronze = cfg.path("bronze")
    silver = cfg.path("silver")
    q = cfg.quality

    files = sorted(bronze.glob(f"{cfg.data.service}_tripdata_*.parquet"))
    if months:
        files = [f for f in files if any(m in f.name for m in months)]
    if not files:
        raise FileNotFoundError(f"no bronze parquet files in {bronze}")

    tags = [f.stem.split("_")[-1] for f in files]
    win_start = f"{min(tags)}-01"
    end_y, end_m = (int(x) for x in max(tags).split("-"))
    win_end = f"{end_y + (end_m == 12)}-{1 if end_m == 12 else end_m + 1:02d}-01"

    params = {
        "min_sec": q.min_trip_seconds,
        "max_sec": q.max_trip_seconds,
        "min_mi": q.min_trip_miles,
        "max_mi": q.max_trip_miles,
        "min_fare": q.min_fare,
        "max_fare": q.max_fare,
        "max_pax": q.max_passengers,
        "win_start": win_start,
        "win_end": win_end,
    }

    def sub(text: str) -> str:
        return substitute(text, params)

    rejects = reject_clause(params)
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={_duckdb_threads(cfg)}")
    globs = [f.as_posix() for f in files]

    out_path = silver / "trips.parquet"
    out_sql = out_path.as_posix()
    report = QualityReport(stage="silver")

    with stage(log, "silver_build", files=len(files)) as st:
        con.execute(
            f"""
            CREATE OR REPLACE VIEW raw AS
            SELECT {_select_expr()}
            FROM read_parquet({globs}, union_by_name=true)
            """
        )
        report.rows_in = con.execute("SELECT count(*) FROM raw").fetchone()[0]

        con.execute(
            """
            CREATE OR REPLACE TABLE deduped AS
            SELECT DISTINCT * FROM raw
            """
        )
        deduped_rows = con.execute("SELECT count(*) FROM deduped").fetchone()[0]
        report.duplicates_removed = report.rows_in - deduped_rows

        for reason, expr in REASON_SQL.items():
            n = con.execute(
                f"SELECT count(*) FROM deduped WHERE COALESCE({sub(expr)}, FALSE)"
            ).fetchone()[0]
            if n:
                report.reasons[reason] = int(n)

        attributed = con.execute(
            f"SELECT count(*) FROM deduped WHERE {rejects}"
        ).fetchone()[0]

        con.execute(
            f"""
            COPY (
                SELECT *,
                       date_trunc('hour', pickup_ts) AS hour_ts,
                       trip_seconds / 60.0           AS trip_minutes
                FROM deduped
                WHERE NOT ({rejects})
            ) TO '{out_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        report.rows_out = con.execute(
            f"SELECT count(*) FROM read_parquet('{out_sql}')"
        ).fetchone()[0]
        report.quarantined = deduped_rows - report.rows_out
        if report.quarantined != attributed:
            raise ValueError(
                f"quality reconciliation failed: {report.quarantined:,} rows were dropped but "
                f"only {attributed:,} match a named rule. A filter predicate is evaluating to "
                f"NULL and silently discarding rows."
            )
        st["rows_in"] = report.rows_in
        st["rows_out"] = report.rows_out
        st["quarantine_rate"] = round(report.quarantine_rate, 4)

    con.close()

    with open(cfg.path("reports") / "quality_silver.json", "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2)
    log_event(log, "silver quality", **report.to_dict())

    if report.quarantine_rate > float(q.max_quarantine_rate):
        breakdown = ", ".join(
            f"{k}={v:,} ({v / report.rows_in:.2%})"
            for k, v in sorted(report.reasons.items(), key=lambda x: -x[1])
        )
        raise ValueError(
            f"quarantine rate {report.quarantine_rate:.3f} exceeds threshold "
            f"{float(q.max_quarantine_rate):.3f}. Rejections by rule: {breakdown}. "
            f"Full report at reports/quality_silver.json. Raise "
            f"quality.max_quarantine_rate only after confirming these rejections are correct."
        )
    return out_path, report


def silver_sample(path: Path, n: int = 100_000) -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute(
        f"SELECT * FROM read_parquet('{Path(path).as_posix()}') USING SAMPLE {n} ROWS"
    ).fetch_df()
    con.close()
    return df
