from __future__ import annotations

import numpy as np
import pytest

from cabcast.optimize.rebalance import (
    build_cost_matrix,
    evaluate_rebalancing,
    exact_transport,
    sinkhorn,
)

pytest.importorskip("ot", reason="exact LP comparison needs POT")


@pytest.fixture(scope="module")
def problem():
    rng = np.random.default_rng(5)
    n = 60
    xy = rng.uniform(0, 30, (n, 2))
    cost = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
    return rng.gamma(2, 1, n), rng.gamma(2, 1, n), cost


def test_plan_marginals_match_supply_and_demand(problem):
    supply, demand, cost = problem
    plan = sinkhorn(supply, demand, cost, epsilon=0.02, max_iter=4000, tol=1e-9)
    assert plan.converged
    assert np.allclose(plan.plan.sum(axis=1), supply / supply.sum(), atol=1e-6)
    assert np.allclose(plan.plan.sum(axis=0), demand / demand.sum(), atol=1e-6)


def test_plan_is_non_negative_and_sums_to_one(problem):
    supply, demand, cost = problem
    plan = sinkhorn(supply, demand, cost, epsilon=0.05, max_iter=2000, tol=1e-9)
    assert (plan.plan >= 0).all()
    assert plan.total_mass == pytest.approx(1.0, abs=1e-6)


def test_entropic_cost_upper_bounds_the_exact_optimum(problem):
    supply, demand, cost = problem
    approx = sinkhorn(supply, demand, cost, epsilon=0.01, max_iter=6000, tol=1e-10)
    exact = exact_transport(supply, demand, cost)
    assert approx.cost >= exact.cost - 1e-9
    assert (approx.cost - exact.cost) / exact.cost < 0.25


def test_smaller_epsilon_moves_cost_toward_the_optimum(problem):
    supply, demand, cost = problem
    exact = exact_transport(supply, demand, cost)
    coarse = sinkhorn(supply, demand, cost, epsilon=0.20, max_iter=4000, tol=1e-10)
    fine = sinkhorn(supply, demand, cost, epsilon=0.01, max_iter=8000, tol=1e-10)
    assert abs(fine.cost - exact.cost) < abs(coarse.cost - exact.cost)


def test_identical_distributions_stay_on_the_diagonal(problem):
    _, _, cost = problem
    uniform = np.ones(cost.shape[0])
    plan = sinkhorn(uniform, uniform, cost, epsilon=0.005, max_iter=8000, tol=1e-10)
    assert np.trace(plan.plan) > 0.85


def test_cost_matrix_penalises_beyond_the_service_radius():
    travel = np.array([[0.0, 10.0, 40.0], [10.0, 0.0, 30.0], [40.0, 30.0, 0.0]])
    cost = build_cost_matrix(travel, service_radius_minutes=25.0, penalty=4.0)
    assert cost[0, 1] == pytest.approx(10.0)
    assert cost[0, 2] > travel[0, 2]
    assert np.allclose(np.diag(cost), 0.0)


def test_zero_mass_input_is_rejected(problem):
    supply, demand, cost = problem
    with pytest.raises(ValueError):
        sinkhorn(np.zeros_like(supply), demand, cost)


def test_rebalancing_reduces_unmet_demand():
    rng = np.random.default_rng(9)
    n = 40
    xy = rng.uniform(0, 25, (n, 2))
    travel = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
    np.fill_diagonal(travel, 0.0)
    demand = rng.gamma(2.5, 4.0, n)
    supply = demand[::-1] + 1.0

    result, plan = evaluate_rebalancing(
        idle_supply=supply, predicted_demand=demand, travel_minutes=travel,
        fleet_size=1000.0, service_radius_minutes=25.0,
        epsilon=0.05, max_iter=2000, tol=1e-9,
    )
    assert plan.converged
    assert result.unmet_after < result.unmet_before
    assert result.unmet_reduction_pct > 0
    assert result.served_after >= result.served_before
