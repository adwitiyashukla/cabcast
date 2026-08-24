from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from urbanflow import __version__
from urbanflow.config import load_config
from urbanflow.geo.graph import build_zone_graph
from urbanflow.geo.zones import get_zones
from urbanflow.logging_utils import get_logger, log_event
from urbanflow.models.conformal import MondrianConformalizedQuantile
from urbanflow.models.registry import ModelRegistry
from urbanflow.optimize.rebalance import evaluate_rebalancing
from urbanflow.serving.schemas import (
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ModelCardResponse,
    Move,
    RebalanceRequest,
    RebalanceResponse,
    ZoneForecast,
)

log = get_logger("urbanflow.serving")


class ServingContext:
    def __init__(self) -> None:
        self.cfg = load_config()
        registry = ModelRegistry(self.cfg.path("artifacts"))
        available = registry.list_models()
        required = ["demand_point", "demand_q_lower", "demand_q_upper"]
        missing = [m for m in required if m not in available]
        if missing:
            raise RuntimeError(
                f"missing models {missing}. Run `urbanflow all` before starting the API."
            )

        self.point, self.point_card, _ = registry.load("demand_point")
        self.lower, _, _ = registry.load("demand_q_lower")
        self.upper, _, _ = registry.load("demand_q_upper")

        store_path = Path(self.cfg.path("artifacts")) / "feature_store.parquet"
        if not store_path.exists():
            raise RuntimeError(f"feature store missing at {store_path}")
        self.store = pd.read_parquet(store_path)
        self.store["hour_ts"] = pd.to_datetime(self.store["hour_ts"])

        conformal_path = Path(self.cfg.path("artifacts")) / "conformal.json"
        payload = json.loads(conformal_path.read_text(encoding="utf-8"))
        self.alpha = float(payload["alpha"])
        self.mondrian = MondrianConformalizedQuantile(self.alpha, len(payload["mondrian_qhat"]))
        self.mondrian.edges_ = np.array(payload["mondrian_edges"], dtype=float)
        self.mondrian.qhat_ = {int(k): float(v) for k, v in payload["mondrian_qhat"].items()}
        self.mondrian.global_qhat_ = float(payload["mondrian_global_qhat"])

        self.zones = get_zones(self.cfg)
        self.graph = build_zone_graph(self.zones, int(self.cfg.geo.laplacian_eigenvectors))
        self.models = available
        log_event(
            log, "serving context ready", models=available, feature_rows=len(self.store)
        )

    def slice_for(self, hour_ts: pd.Timestamp, zone_ids: list[int] | None) -> pd.DataFrame:
        sub = self.store[self.store["hour_ts"] == hour_ts]
        if zone_ids:
            sub = sub[sub["zone_id"].isin(zone_ids)]
        return sub.sort_values("zone_id")

    def predict(self, sub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        yhat = self.point.predict(sub)
        lo = self.lower.predict(sub)
        hi = np.maximum(self.upper.predict(sub), lo)
        lower, upper = self.mondrian.interval(lo, hi, yhat)
        return yhat, lower, upper

    def available_hours(self) -> list[str]:
        hours = np.sort(self.store["hour_ts"].unique())
        return [str(pd.Timestamp(h)) for h in hours[:3]] + [str(pd.Timestamp(hours[-1]))]


@lru_cache(maxsize=1)
def context() -> ServingContext:
    return ServingContext()


app = FastAPI(
    title="UrbanFlow demand API",
    version=__version__,
    description=(
        "Zone-level NYC taxi demand forecasts with conformal prediction intervals, "
        "plus an optimal-transport fleet rebalancing endpoint."
    ),
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ctx = context()
    return HealthResponse(
        status="ok",
        version=__version__,
        models_loaded=ctx.models,
        feature_store_rows=int(len(ctx.store)),
        coverage_target=1.0 - ctx.alpha,
    )


@app.get("/model/card", response_model=ModelCardResponse)
def model_card() -> ModelCardResponse:
    card = context().point_card
    return ModelCardResponse(
        name=card.name,
        stage=card.stage,
        created_utc=card.created_utc,
        git_sha=card.git_sha,
        n_features=card.n_features,
        n_train_rows=card.n_train_rows,
        best_iteration=card.best_iteration,
        data_source=card.data_source,
        metrics=card.metrics,
    )


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    ctx = context()
    started = time.perf_counter()
    ts = pd.Timestamp(req.hour_ts)
    sub = ctx.slice_for(ts, req.zone_ids)
    if sub.empty:
        raise HTTPException(
            status_code=404,
            detail=f"no features for {ts}. Available hours include {ctx.available_hours()}",
        )

    yhat, lower, upper = ctx.predict(sub)
    forecasts = [
        ZoneForecast(
            zone_id=int(z),
            predicted_trips=round(float(p), 4),
            lower=round(float(lo), 4),
            upper=round(float(hi), 4),
            interval_width=round(float(hi - lo), 4),
        )
        for z, p, lo, hi in zip(sub["zone_id"], yhat, lower, upper, strict=True)
    ]
    log_event(
        log, "forecast served", hour=str(ts), zones=len(forecasts),
        ms=round((time.perf_counter() - started) * 1000, 2),
    )
    return ForecastResponse(
        hour_ts=ts,
        coverage_target=1.0 - ctx.alpha,
        method="Mondrian conformalised quantile regression",
        n_zones=len(forecasts),
        total_predicted_trips=round(float(yhat.sum()), 3),
        forecasts=forecasts,
    )


@app.post("/rebalance", response_model=RebalanceResponse)
def rebalance(req: RebalanceRequest) -> RebalanceResponse:
    ctx = context()
    ts = pd.Timestamp(req.hour_ts)
    sub = ctx.slice_for(ts, None)
    if sub.empty:
        raise HTTPException(
            status_code=404,
            detail=f"no features for {ts}. Available hours include {ctx.available_hours()}",
        )

    zone_ids = sub["zone_id"].to_numpy()
    yhat, _, _ = ctx.predict(sub)
    idx = np.searchsorted(ctx.graph.zone_ids, zone_ids)
    travel = ctx.graph.travel_minutes[np.ix_(idx, idx)]

    if req.idle_supply:
        supply = np.array([req.idle_supply.get(int(z), 0.0) for z in zone_ids], dtype=float)
        if supply.sum() <= 0:
            raise HTTPException(status_code=422, detail="idle_supply must carry positive mass")
    else:
        supply = np.full(len(zone_ids), 1.0)

    result, plan = evaluate_rebalancing(
        idle_supply=supply,
        predicted_demand=yhat,
        travel_minutes=travel,
        fleet_size=req.fleet_size,
        service_radius_minutes=req.service_radius_minutes,
        epsilon=float(ctx.cfg.optimize.sinkhorn_epsilon),
        max_iter=int(ctx.cfg.optimize.sinkhorn_max_iter),
        tol=float(ctx.cfg.optimize.sinkhorn_tol),
    )

    moves = result.moves.copy()
    np.fill_diagonal(moves, 0.0)
    flat = np.dstack(np.unravel_index(np.argsort(moves, axis=None)[::-1], moves.shape))[0][:20]
    top = [
        Move(
            from_zone=int(zone_ids[i]),
            to_zone=int(zone_ids[j]),
            vehicles=round(float(moves[i, j]), 3),
            travel_minutes=round(float(travel[i, j]), 2),
        )
        for i, j in flat
        if moves[i, j] > 0.01
    ]
    return RebalanceResponse(
        hour_ts=ts,
        n_zones=len(zone_ids),
        unmet_before=round(result.unmet_before, 3),
        unmet_after=round(result.unmet_after, 3),
        unmet_reduction_pct=round(result.unmet_reduction_pct, 3),
        vehicle_minutes=round(result.vehicle_minutes, 2),
        sinkhorn_iterations=plan.iterations,
        converged=bool(plan.converged),
        top_moves=top,
    )
