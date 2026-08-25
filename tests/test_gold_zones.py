from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from cabcast.data.gold import _active_zones


@pytest.fixture
def silver(tmp_path):
    rows = []
    hours = pd.date_range("2025-01-01", periods=24 * 30, freq="h")
    for zone, per_hour in [(132, 4), (161, 6), (264, 5), (265, 3), (7, 0)]:
        for h in hours:
            for _ in range(per_hour):
                rows.append({"pickup_zone": zone, "pickup_ts": h})
    frame = pd.DataFrame(rows)
    path = tmp_path / "trips.parquet"
    frame.to_parquet(path, index=False)
    return path


def test_zones_without_geometry_are_excluded(silver):
    con = duckdb.connect()
    active, unplaceable = _active_zones(con, silver, 1.0, known_zones=list(range(1, 264)))
    con.close()
    assert 264 not in active
    assert 265 not in active
    assert set(unplaceable) == {264, 265}


def test_busy_known_zones_survive(silver):
    con = duckdb.connect()
    active, _ = _active_zones(con, silver, 1.0, known_zones=list(range(1, 264)))
    con.close()
    assert 132 in active
    assert 161 in active


def test_quiet_zones_are_filtered_by_volume(silver):
    con = duckdb.connect()
    active, _ = _active_zones(con, silver, 200.0, known_zones=list(range(1, 264)))
    con.close()
    assert active == []


def test_every_active_zone_is_known(silver):
    con = duckdb.connect()
    known = list(range(1, 264))
    active, _ = _active_zones(con, silver, 1.0, known_zones=known)
    con.close()
    assert set(active).issubset(set(known))
