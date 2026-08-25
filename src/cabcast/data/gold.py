from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from cabcast.geo.graph import ZoneGraph, graph_feature_table
from cabcast.geo.zones import ZoneSet
from cabcast.logging_utils import get_logger, log_event, stage
from cabcast.quality.contracts import GOLD_PANEL, QualityReport

log = get_logger(__name__)

def _duckdb_threads(cfg) -> int:
    configured = int(cfg.project.get("num_threads", 0))
    return configured if configured > 0 else max(os.cpu_count() or 4, 1)



def _active_zones(
    con: duckdb.DuckDBPyConnection,
    silver_path: Path,
    min_daily: float,
    known_zones: list[int],
) -> tuple[list[int], list[int]]:
    source = Path(silver_path).as_posix()
    rows = con.execute(
        f"""
        WITH per_zone AS (
            SELECT pickup_zone AS zone_id,
                   count(*) AS trips,
                   date_diff('day', min(pickup_ts), max(pickup_ts)) + 1 AS span_days
            FROM read_parquet('{source}')
            GROUP BY 1
        )
        SELECT zone_id FROM per_zone
        WHERE trips::DOUBLE / greatest(span_days, 1) >= {min_daily}
        ORDER BY zone_id
        """
    ).fetchall()
    busy = [int(r[0]) for r in rows]
    known = {int(z) for z in known_zones}
    active = [z for z in busy if z in known]
    unplaceable = [z for z in busy if z not in known]
    if unplaceable:
        log_event(
            log,
            "zones dropped for having no geometry",
            zones=unplaceable,
            note="TLC location ids outside the shapefile, typically 264 and 265 (Unknown, N/A)",
        )
    return active, unplaceable


def build_gold(
    cfg,
    silver_path: Path,
    zones: ZoneSet,
    graph: ZoneGraph,
) -> tuple[Path, QualityReport]:
    gold = cfg.path("gold")
    external = cfg.path("external")
    out_path = gold / "demand_panel.parquet"
    out_sql = out_path.as_posix()
    source = Path(silver_path).as_posix()
    report = QualityReport(stage="gold")

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={_duckdb_threads(cfg)}")

    with stage(log, "gold_build") as st:
        active, unplaceable = _active_zones(
            con, silver_path, float(cfg.geo.min_daily_trips), list(zones.zone_ids)
        )
        if not active:
            raise ValueError("no zones passed the min_daily_trips filter")
        st["zones_without_geometry"] = len(unplaceable)

        con.execute(
            f"""
            CREATE OR REPLACE TABLE outflow AS
            SELECT pickup_zone                       AS zone_id,
                   hour_ts,
                   count(*)                          AS trips,
                   avg(fare_amount)                  AS mean_fare,
                   avg(trip_miles)                   AS mean_miles,
                   avg(trip_minutes)                 AS mean_duration_min,
                   avg(tip_amount / nullif(fare_amount, 0)) AS mean_tip_rate,
                   avg(congestion_surcharge)         AS mean_congestion_surcharge,
                   avg(passenger_count)              AS mean_passengers
            FROM read_parquet('{source}')
            WHERE pickup_zone IN ({",".join(map(str, active))})
            GROUP BY 1, 2
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE TABLE inflow AS
            SELECT dropoff_zone AS zone_id,
                   date_trunc('hour', dropoff_ts) AS hour_ts,
                   count(*) AS arrivals
            FROM read_parquet('{source}')
            WHERE dropoff_zone IN ({",".join(map(str, active))})
            GROUP BY 1, 2
            """
        )
        bounds = con.execute("SELECT min(hour_ts), max(hour_ts) FROM outflow").fetchone()

        weather_path = external / "weather_hourly.parquet"
        weather_sql = weather_path.as_posix()
        if not weather_path.exists():
            raise FileNotFoundError(f"missing weather table at {weather_path}")

        zone_attrs = zones.table()[
            [
                "zone_id",
                "borough",
                "zone_name",
                "area_sqkm",
                "centroid_lon",
                "centroid_lat",
                "km_from_midtown",
                "is_airport",
                "in_crz",
            ]
        ]
        graph_feats = graph_feature_table(graph)
        con.register("zone_attrs", zone_attrs)
        con.register("graph_feats", graph_feats)

        con.execute(
            f"""
            COPY (
                WITH grid AS (
                    SELECT z.zone_id, g.hour_ts
                    FROM (SELECT unnest([{",".join(map(str, active))}]) AS zone_id) z
                    CROSS JOIN (
                        SELECT unnest(generate_series(
                            TIMESTAMP '{bounds[0]}', TIMESTAMP '{bounds[1]}', INTERVAL 1 HOUR
                        )) AS hour_ts
                    ) g
                )
                SELECT grid.zone_id,
                       grid.hour_ts,
                       COALESCE(o.trips, 0)::DOUBLE     AS trips,
                       COALESCE(i.arrivals, 0)::DOUBLE  AS arrivals,
                       o.mean_fare,
                       o.mean_miles,
                       o.mean_duration_min,
                       o.mean_tip_rate,
                       o.mean_congestion_surcharge,
                       o.mean_passengers,
                       w.temperature_2m,
                       w.precipitation,
                       w.snowfall,
                       w.wind_speed_10m,
                       w.relative_humidity_2m,
                       w.cloud_cover,
                       za.* EXCLUDE (zone_id),
                       gf.* EXCLUDE (zone_id)
                FROM grid
                LEFT JOIN outflow o USING (zone_id, hour_ts)
                LEFT JOIN inflow  i USING (zone_id, hour_ts)
                LEFT JOIN read_parquet('{weather_sql}') w ON w.hour_ts = grid.hour_ts
                LEFT JOIN zone_attrs za ON za.zone_id = grid.zone_id
                LEFT JOIN graph_feats gf ON gf.zone_id = grid.zone_id
                ORDER BY grid.zone_id, grid.hour_ts
            ) TO '{out_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        report.rows_out = con.execute(
            f"SELECT count(*) FROM read_parquet('{out_sql}')"
        ).fetchone()[0]
        report.rows_in = con.execute("SELECT count(*) FROM outflow").fetchone()[0]
        st["zones"] = len(active)
        st["rows"] = report.rows_out

    head = con.execute(f"SELECT * FROM read_parquet('{out_sql}') LIMIT 200000").fetch_df()
    con.close()

    attribute_cols = ["borough", "in_crz", "is_airport", "area_sqkm", "km_from_midtown"]
    null_attrs = {c: int(head[c].isna().sum()) for c in attribute_cols if c in head.columns}
    offenders = {c: n for c, n in null_attrs.items() if n}
    if offenders:
        raise ValueError(
            f"gold panel has zones with missing attributes: {offenders}. Every modelled zone "
            f"must join to the zone table."
        )

    report.contract_violations = GOLD_PANEL.validate(head)
    if report.contract_violations:
        raise ValueError(f"gold contract failed: {report.contract_violations}")

    with open(cfg.path("reports") / "quality_gold.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                **report.to_dict(),
                "active_zones": len(active),
                "zones_without_geometry": unplaceable,
            },
            fh,
            indent=2,
        )
    log_event(log, "gold panel written", rows=report.rows_out, zones=len(active))
    return out_path, report


def load_panel(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    con = duckdb.connect()
    cols = ", ".join(columns) if columns else "*"
    source = Path(path).as_posix()
    df = con.execute(
        f"SELECT {cols} FROM read_parquet('{source}') ORDER BY zone_id, hour_ts"
    ).fetch_df()
    con.close()
    return df


def panel_summary(df: pd.DataFrame) -> dict[str, float]:
    trips = df["trips"].to_numpy()
    return {
        "rows": int(len(df)),
        "zones": int(df["zone_id"].nunique()),
        "hours": int(df["hour_ts"].nunique()),
        "total_trips": int(trips.sum()),
        "mean_trips_per_cell": float(np.round(trips.mean(), 3)),
        "zero_cell_share": float(np.round((trips == 0).mean(), 4)),
        "p99_trips": float(np.percentile(trips, 99)),
        "max_trips": float(trips.max()),
    }
