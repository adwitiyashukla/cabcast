from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from urbanflow.quality.contracts import (
    GOLD_PANEL,
    SILVER_TRIPS,
    ColumnSpec,
    QualityReport,
    TableContract,
)


@pytest.fixture
def clean_panel():
    hours = pd.date_range("2025-01-01", periods=48, freq="h")
    return pd.DataFrame(
        {
            "zone_id": np.repeat([1, 2], 48),
            "hour_ts": np.tile(hours, 2),
            "trips": np.arange(96, dtype=float),
            "mean_fare": 18.0,
            "mean_miles": 3.0,
            "mean_duration_min": 12.0,
            "temperature_2m": 8.0,
            "precipitation": 0.0,
        }
    )


def test_clean_panel_passes_the_contract(clean_panel):
    assert GOLD_PANEL.validate(clean_panel) == []


def test_missing_column_is_reported(clean_panel):
    problems = GOLD_PANEL.validate(clean_panel.drop(columns="trips"))
    assert any("missing columns" in p for p in problems)


def test_negative_target_is_reported(clean_panel):
    bad = clean_panel.copy()
    bad.loc[0, "trips"] = -1.0
    assert any("below 0" in p for p in GOLD_PANEL.validate(bad))


def test_duplicate_primary_key_is_reported(clean_panel):
    bad = pd.concat([clean_panel, clean_panel.head(1)], ignore_index=True)
    assert any("duplicate primary keys" in p for p in GOLD_PANEL.validate(bad))


def test_out_of_range_zone_is_reported(clean_panel):
    bad = clean_panel.copy()
    bad.loc[0, "zone_id"] = 9999
    assert any("above 265" in p for p in GOLD_PANEL.validate(bad))


def test_nulls_in_non_nullable_column_are_reported(clean_panel):
    bad = clean_panel.copy()
    bad.loc[0, "trips"] = np.nan
    assert any("nulls" in p for p in GOLD_PANEL.validate(bad))


def test_allowed_values_are_enforced():
    contract = TableContract(
        name="t", columns=(ColumnSpec("payment", "numeric", allowed=(1, 2)),)
    )
    assert contract.validate(pd.DataFrame({"payment": [1, 2]})) == []
    assert contract.validate(pd.DataFrame({"payment": [1, 5]})) != []


def test_silver_contract_covers_the_expected_columns():
    names = SILVER_TRIPS.column_names
    for expected in ("pickup_ts", "dropoff_ts", "pickup_zone", "trip_seconds", "fare_amount"):
        assert expected in names


def test_quality_report_rate_and_serialisation():
    report = QualityReport(stage="silver", rows_in=1000, rows_out=950, quarantined=50)
    assert report.quarantine_rate == pytest.approx(0.05)
    assert report.to_dict()["quarantine_rate"] == pytest.approx(0.05)
