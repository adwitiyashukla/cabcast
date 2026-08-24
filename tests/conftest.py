from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from urbanflow.config import load_config
from urbanflow.geo.graph import build_zone_graph
from urbanflow.geo.zones import build_synthetic_zones


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def zones():
    return build_synthetic_zones(seed=7)


@pytest.fixture(scope="session")
def graph(zones):
    return build_zone_graph(zones, n_eigenvectors=6)


@pytest.fixture(scope="session")
def toy_panel():
    rng = np.random.default_rng(11)
    hours = pd.date_range("2025-01-01", periods=24 * 60, freq="h")
    zone_ids = [1, 2, 3, 4, 5, 6]
    rows = []
    for z in zone_ids:
        level = 4.0 + 3.0 * z
        profile = 1.0 + 0.7 * np.sin(2 * np.pi * (hours.hour - 8) / 24.0)
        weekly = 1.0 + 0.2 * (hours.dayofweek < 5)
        lam = level * profile * weekly
        rows.append(
            pd.DataFrame(
                {
                    "zone_id": z,
                    "hour_ts": hours,
                    "trips": rng.poisson(np.clip(lam, 0.1, None)).astype(float),
                    "arrivals": rng.poisson(np.clip(lam * 0.9, 0.1, None)).astype(float),
                    "mean_fare": rng.normal(18, 3, len(hours)),
                    "mean_miles": rng.normal(3, 0.6, len(hours)),
                    "mean_duration_min": rng.normal(13, 3, len(hours)),
                    "mean_tip_rate": rng.normal(0.2, 0.03, len(hours)),
                    "mean_congestion_surcharge": 2.5,
                    "mean_passengers": 1.4,
                    "temperature_2m": rng.normal(10, 6, len(hours)),
                    "precipitation": np.clip(rng.gamma(0.4, 0.6, len(hours)) - 0.4, 0, None),
                    "snowfall": 0.0,
                    "wind_speed_10m": rng.gamma(2, 3, len(hours)),
                    "relative_humidity_2m": rng.normal(60, 8, len(hours)),
                    "cloud_cover": rng.normal(45, 15, len(hours)),
                    "borough": "Manhattan",
                    "zone_name": f"Zone {z}",
                    "area_sqkm": 2.0 + z * 0.1,
                    "centroid_lon": -73.98 + z * 0.002,
                    "centroid_lat": 40.75 + z * 0.002,
                    "km_from_midtown": float(z),
                    "is_airport": False,
                    "in_crz": z <= 3,
                    "graph_degree": 4.0,
                    "graph_betweenness": 0.02,
                    "graph_closeness": 0.3,
                    "graph_pagerank": 0.004,
                    "graph_eigencentrality": 0.1,
                    **{f"lap_eig_{i}": rng.normal(0, 0.1, len(hours)) for i in range(1, 9)},
                }
            )
        )
    return pd.concat(rows, ignore_index=True)
