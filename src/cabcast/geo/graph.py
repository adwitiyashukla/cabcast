from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.sparse.linalg import eigsh

from cabcast.geo.zones import ZoneSet
from cabcast.logging_utils import get_logger, log_event

log = get_logger(__name__)

FEET_PER_KM = 3280.84
FREEFLOW_KMH = 24.0
MIN_TRAVEL_MINUTES = 2.5


@dataclass(frozen=True)
class ZoneGraph:
    zone_ids: np.ndarray
    adjacency: sp.csr_matrix
    travel_minutes: np.ndarray
    eigenmaps: np.ndarray
    centrality: pd.DataFrame

    @property
    def n(self) -> int:
        return len(self.zone_ids)

    def index_of(self, zone_id: int) -> int:
        return int(np.searchsorted(self.zone_ids, zone_id))

    def transition_matrix(self) -> sp.csr_matrix:
        deg = np.asarray(self.adjacency.sum(axis=1)).ravel()
        deg[deg == 0] = 1.0
        return sp.diags(1.0 / deg) @ self.adjacency


def _touch_adjacency(zones: ZoneSet) -> sp.lil_matrix:
    n = zones.n_zones
    adj = sp.lil_matrix((n, n), dtype=np.float64)
    gdf = zones.gdf.reset_index(drop=True)
    sindex = gdf.sindex
    for i, geom in enumerate(gdf.geometry):
        probe = geom.buffer(1e-6)
        for j in sindex.query(probe):
            j = int(j)
            if j == i:
                continue
            if probe.intersects(gdf.geometry.iloc[j]):
                adj[i, j] = 1.0
                adj[j, i] = 1.0
    return adj


def _connect_components(adj: sp.lil_matrix, xy: np.ndarray) -> sp.lil_matrix:
    n_comp, labels = connected_components(adj.tocsr(), directed=False)
    while n_comp > 1:
        best = None
        for a in range(n_comp):
            ia = np.flatnonzero(labels == a)
            ib = np.flatnonzero(labels != a)
            d = np.linalg.norm(xy[ia][:, None, :] - xy[ib][None, :, :], axis=2)
            k = np.unravel_index(np.argmin(d), d.shape)
            cand = (d[k], int(ia[k[0]]), int(ib[k[1]]))
            if best is None or cand[0] < best[0]:
                best = cand
        _, i, j = best
        adj[i, j] = 1.0
        adj[j, i] = 1.0
        n_comp, labels = connected_components(adj.tocsr(), directed=False)
    return adj


def _laplacian_eigenmaps(adj: sp.csr_matrix, k: int) -> np.ndarray:
    n = adj.shape[0]
    deg = np.asarray(adj.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(deg))
    lap = sp.identity(n, format="csr") - d_inv_sqrt @ adj @ d_inv_sqrt
    k_eff = min(k + 1, n - 1)
    vals, vecs = eigsh(lap.astype(np.float64), k=k_eff, sigma=-1e-3, which="LM")
    order = np.argsort(vals)
    vecs = vecs[:, order][:, 1 : k + 1]
    for c in range(vecs.shape[1]):
        col = vecs[:, c]
        pivot = col[np.argmax(np.abs(col))]
        if pivot < 0:
            vecs[:, c] = -col
    if vecs.shape[1] < k:
        vecs = np.pad(vecs, ((0, 0), (0, k - vecs.shape[1])))
    return vecs


def _centrality(adj: sp.csr_matrix, zone_ids: np.ndarray) -> pd.DataFrame:
    g = nx.from_scipy_sparse_array(adj)
    deg = np.asarray(adj.sum(axis=1)).ravel()
    btw = nx.betweenness_centrality(g, normalized=True)
    clo = nx.closeness_centrality(g)
    pr = nx.pagerank(g, alpha=0.85)
    try:
        eig = nx.eigenvector_centrality_numpy(g)
    except Exception:
        eig = dict.fromkeys(g.nodes, 0.0)
    return pd.DataFrame(
        {
            "zone_id": zone_ids,
            "graph_degree": deg,
            "graph_betweenness": [btw[i] for i in range(len(zone_ids))],
            "graph_closeness": [clo[i] for i in range(len(zone_ids))],
            "graph_pagerank": [pr[i] for i in range(len(zone_ids))],
            "graph_eigencentrality": [eig[i] for i in range(len(zone_ids))],
        }
    )


def _travel_minutes(adj: sp.csr_matrix, xy: np.ndarray) -> np.ndarray:
    rows, cols = adj.nonzero()
    dist_km = np.linalg.norm(xy[rows] - xy[cols], axis=1) / FEET_PER_KM
    minutes = np.maximum(dist_km / FREEFLOW_KMH * 60.0, MIN_TRAVEL_MINUTES)
    weighted = sp.csr_matrix((minutes, (rows, cols)), shape=adj.shape)
    tt = shortest_path(weighted, method="D", directed=False)
    euclid_km = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2) / FEET_PER_KM
    fallback = np.maximum(euclid_km / FREEFLOW_KMH * 60.0 * 1.4, MIN_TRAVEL_MINUTES)
    tt = np.where(np.isfinite(tt), tt, fallback)
    np.fill_diagonal(tt, 0.0)
    return tt


def build_zone_graph(zones: ZoneSet, n_eigenvectors: int = 8) -> ZoneGraph:
    xy = zones.centroids_xy()
    adj = _touch_adjacency(zones)
    adj = _connect_components(adj, xy)
    adj_csr = adj.tocsr()
    adj_csr.eliminate_zeros()

    eigenmaps = _laplacian_eigenmaps(adj_csr, n_eigenvectors)
    centrality = _centrality(adj_csr, zones.zone_ids)
    travel = _travel_minutes(adj_csr, xy)

    deg = np.asarray(adj_csr.sum(axis=1)).ravel()
    log_event(
        log,
        "zone graph built",
        nodes=zones.n_zones,
        edges=int(adj_csr.nnz // 2),
        mean_degree=round(float(deg.mean()), 2),
        eigenvectors=int(eigenmaps.shape[1]),
        median_travel_minutes=round(float(np.median(travel[travel > 0])), 1),
    )
    return ZoneGraph(
        zone_ids=zones.zone_ids,
        adjacency=adj_csr,
        travel_minutes=travel,
        eigenmaps=eigenmaps,
        centrality=centrality,
    )


def graph_feature_table(graph: ZoneGraph) -> pd.DataFrame:
    eig = pd.DataFrame(
        graph.eigenmaps,
        columns=[f"lap_eig_{i + 1}" for i in range(graph.eigenmaps.shape[1])],
    )
    eig.insert(0, "zone_id", graph.zone_ids)
    return graph.centrality.merge(eig, on="zone_id", how="inner")
