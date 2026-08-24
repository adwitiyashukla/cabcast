from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import connected_components

from urbanflow.geo.zones import (
    BOROUGH_ZONE_COUNTS,
    N_ZONES,
    find_shapefile,
    get_zones,
    zone_h3_cells,
)


def test_zone_count_matches_tlc(zones):
    assert zones.n_zones == N_ZONES == 263


def test_every_borough_present(zones):
    counts = zones.gdf["borough"].value_counts().to_dict()
    for borough, expected in BOROUGH_ZONE_COUNTS.items():
        assert counts[borough] == expected


def test_geometries_are_valid_and_non_empty(zones):
    assert zones.gdf.geometry.is_valid.all()
    assert (zones.gdf["area_sqkm"] > 0).all()


def test_congestion_zone_is_a_manhattan_subset(zones):
    crz = zones.gdf[zones.gdf["in_crz"]]
    assert len(crz) > 0
    assert set(crz["borough"]) == {"Manhattan"}
    assert len(crz) < (zones.gdf["borough"] == "Manhattan").sum()


def test_graph_is_connected(graph):
    n_components, _ = connected_components(graph.adjacency, directed=False)
    assert n_components == 1


def test_adjacency_is_symmetric_and_hollow(graph):
    assert (abs(graph.adjacency - graph.adjacency.T)).nnz == 0
    assert graph.adjacency.diagonal().sum() == 0


def test_travel_matrix_is_a_metric(graph):
    tt = graph.travel_minutes
    assert np.isfinite(tt).all()
    assert np.allclose(np.diag(tt), 0.0)
    assert np.allclose(tt, tt.T)
    i, j, k = 0, 5, 9
    assert tt[i, k] <= tt[i, j] + tt[j, k] + 1e-6


def test_eigenmaps_shape_and_determinism(zones, graph):
    from urbanflow.geo.graph import build_zone_graph

    assert graph.eigenmaps.shape == (zones.n_zones, 6)
    again = build_zone_graph(zones, n_eigenvectors=6)
    assert np.allclose(graph.eigenmaps, again.eigenmaps, atol=1e-8)


def test_h3_covers_every_zone(zones):
    cells = zone_h3_cells(zones, resolution=8)
    assert set(cells["zone_id"]) == set(zones.zone_ids)
    assert cells["h3_cell"].str.len().eq(15).all()


def test_transition_matrix_rows_sum_to_one(graph):
    p = graph.transition_matrix()
    assert np.allclose(np.asarray(p.sum(axis=1)).ravel(), 1.0)


def test_shapefile_is_found_when_nested(tmp_path):
    nested = tmp_path / "taxi_zones" / "taxi_zones"
    nested.mkdir(parents=True)
    (nested / "taxi_zones.shp").write_bytes(b"")
    assert find_shapefile(tmp_path / "taxi_zones") is not None


def test_shapefile_is_found_when_flat(tmp_path):
    flat = tmp_path / "taxi_zones"
    flat.mkdir()
    (flat / "taxi_zones.shp").write_bytes(b"")
    assert find_shapefile(flat) is not None


def test_missing_shapefile_returns_none(tmp_path):
    empty = tmp_path / "taxi_zones"
    empty.mkdir()
    assert find_shapefile(empty) is None


def test_fallback_to_synthetic_is_logged_loudly(cfg, caplog, monkeypatch, tmp_path):
    monkeypatch.setattr("urbanflow.config.Config.path", lambda self, key: tmp_path)
    with caplog.at_level("WARNING"):
        zones = get_zones(cfg, prefer_real=True)
    assert zones.source == "synthetic_voronoi"
    assert any("FALLING BACK TO SYNTHETIC GEOMETRY" in r.message for r in caplog.records)
