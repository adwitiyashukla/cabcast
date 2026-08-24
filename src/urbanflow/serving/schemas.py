from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: list[str]
    feature_store_rows: int
    coverage_target: float


class ForecastRequest(BaseModel):
    hour_ts: datetime = Field(description="the hour to forecast, must exist in the feature store")
    zone_ids: list[int] | None = Field(default=None, description="omit to score every zone")


class ZoneForecast(BaseModel):
    zone_id: int
    predicted_trips: float
    lower: float
    upper: float
    interval_width: float


class ForecastResponse(BaseModel):
    hour_ts: datetime
    coverage_target: float
    method: str
    n_zones: int
    total_predicted_trips: float
    forecasts: list[ZoneForecast]


class RebalanceRequest(BaseModel):
    hour_ts: datetime
    fleet_size: float = Field(default=2500.0, gt=0)
    idle_supply: dict[int, float] | None = Field(
        default=None, description="idle vehicles per zone, defaults to a uniform fleet"
    )
    service_radius_minutes: float = Field(default=25.0, gt=0)


class RebalanceMove:
    pass


class Move(BaseModel):
    from_zone: int
    to_zone: int
    vehicles: float
    travel_minutes: float


class RebalanceResponse(BaseModel):
    hour_ts: datetime
    n_zones: int
    unmet_before: float
    unmet_after: float
    unmet_reduction_pct: float
    vehicle_minutes: float
    sinkhorn_iterations: int
    converged: bool
    top_moves: list[Move]


class ModelCardResponse(BaseModel):
    name: str
    stage: str
    created_utc: str
    git_sha: str
    n_features: int
    n_train_rows: int
    best_iteration: int
    data_source: str
    metrics: dict[str, float]
