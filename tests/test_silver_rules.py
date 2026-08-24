from __future__ import annotations

import duckdb
import pytest

from urbanflow.data.silver import REASON_SQL, reject_clause, substitute

PARAMS = {
    "min_sec": 30,
    "max_sec": 21600,
    "min_mi": 0.05,
    "max_mi": 150.0,
    "min_fare": 0.0,
    "max_fare": 2000.0,
    "max_pax": 9,
    "win_start": "2025-01-01",
    "win_end": "2025-03-01",
}

ROWS = [
    ("clean", "2025-01-05 08:00:00", "2025-01-05 08:12:00", 1, 2.4, 14.0, 18.0, 2.0, 132, 161),
    ("null_passengers", "2025-01-05 08:00:00", "2025-01-05 08:12:00", None, 2.4, 14.0, 18.0, 2.0, 132, 161),
    ("null_distance", "2025-01-05 08:00:00", "2025-01-05 08:12:00", 1, None, 14.0, 18.0, 2.0, 132, 161),
    ("null_fare", "2025-01-05 08:00:00", "2025-01-05 08:12:00", 1, 2.4, None, 18.0, 2.0, 132, 161),
    ("null_zone", "2025-01-05 08:00:00", "2025-01-05 08:12:00", 1, 2.4, 14.0, 18.0, 2.0, None, 161),
    ("null_times", None, None, 1, 2.4, 14.0, 18.0, 2.0, 132, 161),
    ("zero_distance", "2025-01-05 08:00:00", "2025-01-05 08:12:00", 1, 0.0, 14.0, 18.0, 2.0, 132, 161),
    ("negative_fare", "2025-01-05 08:00:00", "2025-01-05 08:12:00", 1, 2.4, -14.0, -18.0, 0.0, 132, 161),
    ("too_short", "2025-01-05 08:00:00", "2025-01-05 08:00:10", 1, 2.4, 14.0, 18.0, 2.0, 132, 161),
    ("too_many_pax", "2025-01-05 08:00:00", "2025-01-05 08:12:00", 12, 2.4, 14.0, 18.0, 2.0, 132, 161),
    ("out_of_window", "2024-06-05 08:00:00", "2024-06-05 08:12:00", 1, 2.4, 14.0, 18.0, 2.0, 132, 161),
]


@pytest.fixture(scope="module")
def con():
    connection = duckdb.connect()
    values = ",\n".join(
        "("
        + ", ".join(
            "NULL" if v is None else (f"'{v}'" if isinstance(v, str) else str(v)) for v in row
        )
        + ")"
        for row in ROWS
    )
    connection.execute(
        f"""
        CREATE TABLE deduped AS
        SELECT label,
               CAST(pu AS TIMESTAMP) AS pickup_ts,
               CAST(dof AS TIMESTAMP) AS dropoff_ts,
               CAST(pax AS DOUBLE)   AS passenger_count,
               CAST(mi AS DOUBLE)    AS trip_miles,
               CAST(fare AS DOUBLE)  AS fare_amount,
               CAST(total AS DOUBLE) AS total_amount,
               CAST(tip AS DOUBLE)   AS tip_amount,
               CAST(pz AS INTEGER)   AS pickup_zone,
               CAST(dz AS INTEGER)   AS dropoff_zone,
               date_diff('second', CAST(pu AS TIMESTAMP), CAST(dof AS TIMESTAMP)) AS trip_seconds
        FROM (VALUES
        {values}
        ) AS v(label, pu, dof, pax, mi, fare, total, tip, pz, dz)
        """
    )
    yield connection
    connection.close()


def _kept(con) -> set[str]:
    rows = con.execute(
        f"SELECT label FROM deduped WHERE NOT ({reject_clause(PARAMS)})"
    ).fetchall()
    return {r[0] for r in rows}


def test_every_dropped_row_is_attributed_to_a_rule(con):
    total = con.execute("SELECT count(*) FROM deduped").fetchone()[0]
    kept = con.execute(
        f"SELECT count(*) FROM deduped WHERE NOT ({reject_clause(PARAMS)})"
    ).fetchone()[0]
    attributed = con.execute(
        f"SELECT count(*) FROM deduped WHERE {reject_clause(PARAMS)}"
    ).fetchone()[0]
    assert kept + attributed == total


def test_null_passenger_count_is_not_a_rejection(con):
    assert "null_passengers" in _kept(con)


def test_clean_row_survives(con):
    assert "clean" in _kept(con)


@pytest.mark.parametrize(
    "label",
    ["null_distance", "null_fare", "null_zone", "null_times", "zero_distance",
     "negative_fare", "too_short", "too_many_pax", "out_of_window"],
)
def test_bad_rows_are_rejected(con, label):
    assert label not in _kept(con)


def test_no_rule_expression_can_evaluate_to_null(con):
    for name, expr in REASON_SQL.items():
        nulls = con.execute(
            f"SELECT count(*) FROM deduped WHERE ({substitute(expr, PARAMS)}) IS NULL"
        ).fetchone()[0]
        assert nulls == 0, f"rule {name} evaluates to NULL on {nulls} rows"


def test_each_rejected_row_matches_at_least_one_named_rule(con):
    for label in ["null_distance", "null_fare", "null_zone", "zero_distance", "too_many_pax"]:
        hits = [
            name
            for name, expr in REASON_SQL.items()
            if con.execute(
                f"SELECT count(*) FROM deduped WHERE label = '{label}' "
                f"AND COALESCE({substitute(expr, PARAMS)}, FALSE)"
            ).fetchone()[0]
        ]
        assert hits, f"{label} was dropped without matching any rule"
