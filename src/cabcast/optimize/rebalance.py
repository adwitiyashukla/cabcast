from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cabcast.logging_utils import get_logger, log_event

log = get_logger(__name__)


@dataclass
class TransportPlan:
    plan: np.ndarray
    cost: float
    iterations: int
    converged: bool
    marginal_error: float

    @property
    def total_mass(self) -> float:
        return float(self.plan.sum())


def sinkhorn(
    supply: np.ndarray,
    demand: np.ndarray,
    cost: np.ndarray,
    epsilon: float = 0.05,
    max_iter: int = 2000,
    tol: float = 1e-9,
) -> TransportPlan:
    a = np.asarray(supply, dtype=np.float64)
    b = np.asarray(demand, dtype=np.float64)
    if a.sum() <= 0 or b.sum() <= 0:
        raise ValueError("supply and demand must both carry positive mass")
    a = a / a.sum()
    b = b / b.sum()

    scale = np.max(np.abs(cost))
    c = cost / scale if scale > 0 else cost
    log_k = -c / epsilon

    f = np.zeros_like(a)
    g = np.zeros_like(b)
    log_a = np.log(np.maximum(a, 1e-300))
    log_b = np.log(np.maximum(b, 1e-300))

    converged = False
    iterations = 0
    while iterations < max_iter:
        iterations += 1
        f = epsilon * (log_a - _logsumexp(log_k + g[None, :] / epsilon, axis=1))
        g = epsilon * (log_b - _logsumexp(log_k + f[:, None] / epsilon, axis=0))
        plan = np.exp(log_k + f[:, None] / epsilon + g[None, :] / epsilon)
        if float(np.abs(plan.sum(axis=1) - a).max()) < tol:
            converged = True
            break

    plan = np.exp(log_k + f[:, None] / epsilon + g[None, :] / epsilon)
    marginal_error = float(
        max(np.abs(plan.sum(axis=1) - a).max(), np.abs(plan.sum(axis=0) - b).max())
    )
    total_cost = float((plan * cost).sum())
    return TransportPlan(plan, total_cost, iterations, converged, marginal_error)


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def exact_transport(supply: np.ndarray, demand: np.ndarray, cost: np.ndarray) -> TransportPlan:
    try:
        import ot
    except ImportError as exc:
        raise ImportError(
            "exact_transport needs the POT package, installed with the dev extras: "
            'pip install -e ".[dev]"'
        ) from exc

    a = np.asarray(supply, dtype=np.float64)
    b = np.asarray(demand, dtype=np.float64)
    a = a / a.sum()
    b = b / b.sum()
    plan = ot.emd(a, b, np.ascontiguousarray(cost, dtype=np.float64), numItermax=200000)
    return TransportPlan(
        plan=plan,
        cost=float((plan * cost).sum()),
        iterations=0,
        converged=True,
        marginal_error=float(
            max(np.abs(plan.sum(axis=1) - a).max(), np.abs(plan.sum(axis=0) - b).max())
        ),
    )


def build_cost_matrix(
    travel_minutes: np.ndarray, service_radius_minutes: float, penalty: float = 4.0
) -> np.ndarray:
    cost = np.array(travel_minutes, dtype=np.float64, copy=True)
    beyond = cost > service_radius_minutes
    cost[beyond] = service_radius_minutes + penalty * (cost[beyond] - service_radius_minutes)
    return cost


@dataclass
class RebalanceResult:
    moves: np.ndarray
    unmet_before: float
    unmet_after: float
    vehicle_minutes: float
    served_before: float
    served_after: float
    vehicles_moved: float
    vehicles_stranded: float

    @property
    def unmet_reduction_pct(self) -> float:
        if self.unmet_before <= 0:
            return 0.0
        return float((self.unmet_before - self.unmet_after) / self.unmet_before * 100.0)

    @property
    def minutes_per_extra_trip(self) -> float:
        gained = self.served_after - self.served_before
        return float(self.vehicle_minutes / gained) if gained > 1e-6 else float("nan")


def evaluate_rebalancing(
    idle_supply: np.ndarray,
    predicted_demand: np.ndarray,
    travel_minutes: np.ndarray,
    fleet_size: float,
    service_radius_minutes: float,
    epsilon: float,
    max_iter: int,
    tol: float,
    reposition_share: float = 0.35,
    horizon_minutes: float = 30.0,
) -> tuple[RebalanceResult, TransportPlan]:
    supply = np.maximum(np.asarray(idle_supply, dtype=float), 0.0)
    demand = np.maximum(np.asarray(predicted_demand, dtype=float), 0.0)
    if supply.sum() <= 0 or demand.sum() <= 0:
        raise ValueError("both supply and demand must be positive")

    supply_units = supply / supply.sum() * fleet_size
    demand_units = demand / demand.sum() * fleet_size

    movable = supply_units * float(reposition_share)
    stationary = supply_units - movable
    deficit = np.maximum(demand_units - stationary, 0.0)
    if deficit.sum() <= 0:
        deficit = demand_units.copy()

    cost = build_cost_matrix(travel_minutes, service_radius_minutes)
    plan = sinkhorn(movable, deficit, cost, epsilon=epsilon, max_iter=max_iter, tol=tol)
    moves = plan.plan * movable.sum()

    reachable = travel_minutes <= float(horizon_minutes)
    arrivals = (moves * reachable).sum(axis=0)
    stranded = (moves * ~reachable).sum(axis=1)
    supply_after = stationary + arrivals + stranded

    served_before = float(np.minimum(supply_units, demand_units).sum())
    served_after = float(np.minimum(supply_after, demand_units).sum())
    unmet_before = float(np.maximum(demand_units - supply_units, 0.0).sum())
    unmet_after = float(np.maximum(demand_units - supply_after, 0.0).sum())

    effective = moves * reachable
    np.fill_diagonal(effective, 0.0)
    vehicle_minutes = float((effective * travel_minutes).sum())

    result = RebalanceResult(
        moves=effective,
        unmet_before=unmet_before,
        unmet_after=unmet_after,
        vehicle_minutes=vehicle_minutes,
        served_before=served_before,
        served_after=served_after,
        vehicles_moved=float(effective.sum()),
        vehicles_stranded=float(stranded.sum()),
    )
    log_event(
        log,
        "rebalance evaluated",
        unmet_before=round(unmet_before, 1),
        unmet_after=round(unmet_after, 1),
        reduction_pct=round(result.unmet_reduction_pct, 2),
        vehicles_moved=round(result.vehicles_moved, 1),
        stranded=round(result.vehicles_stranded, 1),
        sinkhorn_iterations=plan.iterations,
        converged=plan.converged,
    )
    return result, plan
