from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str
    nullable: bool = False
    minimum: float | None = None
    maximum: float | None = None
    allowed: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class TableContract:
    name: str
    columns: tuple[ColumnSpec, ...]
    primary_key: tuple[str, ...] = ()
    unique: tuple[str, ...] = ()

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def validate(self, df: pd.DataFrame) -> list[str]:
        problems: list[str] = []
        missing = [c for c in self.column_names if c not in df.columns]
        if missing:
            problems.append(f"{self.name}: missing columns {missing}")
            return problems

        for spec in self.columns:
            series = df[spec.name]
            if not spec.nullable and series.isna().any():
                problems.append(f"{self.name}.{spec.name}: {int(series.isna().sum())} nulls")
            if spec.dtype == "numeric" and not pd.api.types.is_numeric_dtype(series):
                problems.append(f"{self.name}.{spec.name}: expected numeric, got {series.dtype}")
            if spec.dtype == "datetime" and not pd.api.types.is_datetime64_any_dtype(series):
                problems.append(f"{self.name}.{spec.name}: expected datetime, got {series.dtype}")
            if spec.minimum is not None and pd.api.types.is_numeric_dtype(series):
                bad = int((series.dropna() < spec.minimum).sum())
                if bad:
                    problems.append(f"{self.name}.{spec.name}: {bad} below {spec.minimum}")
            if spec.maximum is not None and pd.api.types.is_numeric_dtype(series):
                bad = int((series.dropna() > spec.maximum).sum())
                if bad:
                    problems.append(f"{self.name}.{spec.name}: {bad} above {spec.maximum}")
            if spec.allowed is not None:
                bad = int((~series.dropna().isin(spec.allowed)).sum())
                if bad:
                    problems.append(f"{self.name}.{spec.name}: {bad} outside {spec.allowed}")

        if self.primary_key:
            dupes = int(df.duplicated(subset=list(self.primary_key)).sum())
            if dupes:
                problems.append(f"{self.name}: {dupes} duplicate primary keys {self.primary_key}")
        return problems


SILVER_TRIPS = TableContract(
    name="silver_trips",
    columns=(
        ColumnSpec("pickup_ts", "datetime"),
        ColumnSpec("dropoff_ts", "datetime"),
        ColumnSpec("pickup_zone", "numeric", minimum=1, maximum=265),
        ColumnSpec("dropoff_zone", "numeric", minimum=1, maximum=265),
        ColumnSpec("trip_seconds", "numeric", minimum=0),
        ColumnSpec("trip_miles", "numeric", minimum=0),
        ColumnSpec("fare_amount", "numeric", minimum=0),
        ColumnSpec("total_amount", "numeric", minimum=0),
        ColumnSpec("tip_amount", "numeric", minimum=0),
        ColumnSpec("passenger_count", "numeric", nullable=True, minimum=0),
    ),
)

GOLD_PANEL = TableContract(
    name="gold_demand_panel",
    columns=(
        ColumnSpec("zone_id", "numeric", minimum=1, maximum=265),
        ColumnSpec("hour_ts", "datetime"),
        ColumnSpec("trips", "numeric", minimum=0),
        ColumnSpec("mean_fare", "numeric", nullable=True, minimum=0),
        ColumnSpec("mean_miles", "numeric", nullable=True, minimum=0),
        ColumnSpec("mean_duration_min", "numeric", nullable=True, minimum=0),
        ColumnSpec("temperature_2m", "numeric"),
        ColumnSpec("precipitation", "numeric", minimum=0),
    ),
    primary_key=("zone_id", "hour_ts"),
)


@dataclass
class QualityReport:
    stage: str
    rows_in: int = 0
    rows_out: int = 0
    quarantined: int = 0
    duplicates_removed: int = 0
    reasons: dict[str, int] = field(default_factory=dict)
    contract_violations: list[str] = field(default_factory=list)

    @property
    def quarantine_rate(self) -> float:
        return self.quarantined / self.rows_in if self.rows_in else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "quarantined": self.quarantined,
            "duplicates_removed": self.duplicates_removed,
            "quarantine_rate": round(self.quarantine_rate, 5),
            "reasons": self.reasons,
            "contract_violations": self.contract_violations,
        }
