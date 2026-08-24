from __future__ import annotations

import numpy as np
import pandas as pd

from urbanflow.monitoring.drift import (
    feature_drift,
    population_stability_index,
    prediction_drift,
)


def test_psi_is_near_zero_for_identical_samples():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 20000)
    b = rng.normal(0, 1, 20000)
    assert population_stability_index(a, b) < 0.01


def test_psi_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 20000)
    values = [population_stability_index(ref, rng.normal(s, 1, 20000)) for s in (0.1, 0.4, 1.0)]
    assert values == sorted(values)


def test_psi_returns_nan_for_tiny_samples():
    assert np.isnan(population_stability_index(np.arange(3.0), np.arange(3.0)))


def test_feature_drift_flags_only_the_shifted_column():
    rng = np.random.default_rng(1)
    n = 8000
    ref = pd.DataFrame({"stable": rng.normal(0, 1, n), "shifted": rng.normal(0, 1, n)})
    cur = pd.DataFrame({"stable": rng.normal(0, 1, n), "shifted": rng.normal(2.5, 1, n)})

    report = feature_drift(ref, cur, ["stable", "shifted"], 0.1, 0.25, 0.01)
    row = report.table.set_index("feature")
    assert row.loc["shifted", "status"] == "alert"
    assert row.loc["stable", "status"] == "ok"
    assert report.n_alert == 1


def test_prediction_drift_reports_status():
    rng = np.random.default_rng(2)
    same = prediction_drift(rng.normal(5, 1, 6000), rng.normal(5, 1, 6000), 0.25)
    moved = prediction_drift(rng.normal(5, 1, 6000), rng.normal(9, 1, 6000), 0.25)
    assert same["status"] == "ok"
    assert moved["status"] == "alert"
    assert moved["psi"] > same["psi"]


def test_drift_report_serialises():
    rng = np.random.default_rng(3)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 2000)})
    cur = pd.DataFrame({"a": rng.normal(0, 1, 2000)})
    payload = feature_drift(ref, cur, ["a"], 0.1, 0.25, 0.01).to_dict()
    assert payload["n_features"] == 1
    assert "worst" in payload
