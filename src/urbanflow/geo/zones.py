from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from urbanflow.logging_utils import get_logger, log_event

log = get_logger(__name__)

WGS84 = "EPSG:4326"
NY_PLANE = "EPSG:2263"

BOROUGH_ZONE_COUNTS: dict[str, int] = {
    "Manhattan": 69,
    "Brooklyn": 61,
    "Queens": 69,
    "Bronx": 43,
    "Staten Island": 20,
    "EWR": 1,
}
N_ZONES = sum(BOROUGH_ZONE_COUNTS.values())

BOROUGH_BBOX: dict[str, tuple[float, float, float, float]] = {
    "Manhattan": (-74.020, 40.701, -73.907, 40.879),
    "Brooklyn": (-74.042, 40.570, -73.833, 40.739),
    "Queens": (-73.962, 40.541, -73.700, 40.801),
    "Bronx": (-73.933, 40.785, -73.765, 40.917),
    "Staten Island": (-74.259, 40.496, -74.049, 40.651),
    "EWR": (-74.192, 40.682, -74.169, 40.701),
}

BOROUGH_DEMAND_WEIGHT: dict[str, float] = {
    "Manhattan": 1.00,
    "Queens": 0.115,
    "Brooklyn": 0.055,
    "Bronx": 0.012,
    "Staten Island": 0.002,
    "EWR": 0.030,
}

MIDTOWN_LON, MIDTOWN_LAT = -73.9840, 40.7549
CRZ_LAT_CUTOFF = 40.7644


@dataclass(frozen=True)
class ZoneSet:
    gdf: gpd.GeoDataFrame
    source: str

    @property
    def zone_ids(self) -> np.ndarray:
        return self.gdf["zone_id"].to_numpy()

    @property
    def n_zones(self) -> int:
        return len(self.gdf)

    def centroids_xy(self) -> np.ndarray:
        pts = self.gdf.to_crs(NY_PLANE).geometry.centroid
        return np.c_[pts.x.to_numpy(), pts.y.to_numpy()]

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.gdf.drop(columns="geometry"))


def _bounded_voronoi(points: np.ndarray, boundary: Polygon) -> list[Polygon]:
    from scipy.spatial import Voronoi

    minx, miny, maxx, maxy = boundary.bounds
    mirrored = np.vstack(
        [
            points,
            np.c_[2 * minx - points[:, 0], points[:, 1]],
            np.c_[2 * maxx - points[:, 0], points[:, 1]],
            np.c_[points[:, 0], 2 * miny - points[:, 1]],
            np.c_[points[:, 0], 2 * maxy - points[:, 1]],
        ]
    )
    vor = Voronoi(mirrored)
    cells: list[Polygon] = []
    for i in range(len(points)):
        region = vor.regions[vor.point_region[i]]
        if not region or -1 in region:
            cells.append(boundary.centroid.buffer(0.001))
            continue
        poly = Polygon([vor.vertices[j] for j in region])
        clipped = poly.intersection(boundary)
        if clipped.is_empty:
            clipped = boundary.centroid.buffer(0.001)
        if clipped.geom_type == "MultiPolygon":
            clipped = max(clipped.geoms, key=lambda g: g.area)
        cells.append(clipped)
    return cells


def build_synthetic_zones(seed: int = 20260823) -> ZoneSet:
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    geoms: list[Polygon] = []
    zone_id = 1

    for borough, count in BOROUGH_ZONE_COUNTS.items():
        minx, miny, maxx, maxy = BOROUGH_BBOX[borough]
        boundary = box(minx, miny, maxx, maxy)
        if count == 1:
            cells = [boundary]
        else:
            pts = np.c_[rng.uniform(minx, maxx, count), rng.uniform(miny, maxy, count)]
            for _ in range(6):
                cells = _bounded_voronoi(pts, boundary)
                pts = np.array([[c.centroid.x, c.centroid.y] for c in cells])
            cells = _bounded_voronoi(pts, boundary)

        for k, cell in enumerate(cells):
            records.append(
                {
                    "zone_id": zone_id,
                    "zone_name": f"{borough} Zone {k + 1:03d}",
                    "borough": borough,
                    "service_zone": "EWR"
                    if borough == "EWR"
                    else ("Yellow Zone" if borough == "Manhattan" else "Boro Zone"),
                }
            )
            geoms.append(cell)
            zone_id += 1

    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs=WGS84)
    gdf = _augment(gdf)
    log_event(log, "synthetic zones built", n_zones=len(gdf), seed=seed)
    return ZoneSet(gdf=gdf, source="synthetic_voronoi")


def find_shapefile(shapefile_dir: Path) -> Path | None:
    matches = sorted(Path(shapefile_dir).rglob("*.shp"))
    return matches[0] if matches else None


def load_tlc_zones(shapefile_dir: Path, lookup_csv: Path) -> ZoneSet:
    shp = find_shapefile(shapefile_dir)
    if shp is None:
        raise FileNotFoundError(f"no .shp found anywhere under {shapefile_dir}")
    gdf = gpd.read_file(shp).to_crs(WGS84)
    lookup = pd.read_csv(lookup_csv)

    gdf = gdf.rename(columns={"LocationID": "zone_id"})
    lookup = lookup.rename(
        columns={
            "LocationID": "zone_id",
            "Zone": "zone_name",
            "Borough": "borough",
        }
    )
    gdf = gdf[["zone_id", "geometry"]].merge(
        lookup[["zone_id", "zone_name", "borough", "service_zone"]], on="zone_id", how="left"
    )
    gdf = gdf.dissolve(by="zone_id", aggfunc="first").reset_index()
    for col in ("zone_name", "borough", "service_zone"):
        gdf[col] = gdf[col].fillna("Unknown")
    gdf = _augment(gdf)
    log_event(log, "TLC zones loaded", n_zones=len(gdf), shapefile=shp.name)
    return ZoneSet(gdf=gdf, source="tlc_shapefile")


def _augment(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    projected = gdf.to_crs(NY_PLANE)
    gdf = gdf.copy()
    gdf["area_sqkm"] = projected.geometry.area.to_numpy() / 1e6 * 0.09290304
    cent = projected.geometry.centroid.to_crs(WGS84)
    gdf["centroid_lon"] = cent.x.to_numpy()
    gdf["centroid_lat"] = cent.y.to_numpy()

    gdf["is_airport"] = gdf["zone_name"].str.contains("Airport", case=False, na=False)
    gdf["in_crz"] = (gdf["borough"] == "Manhattan") & (gdf["centroid_lat"] < CRZ_LAT_CUTOFF)

    gdf["km_from_midtown"] = np.hypot(
        (gdf["centroid_lon"].to_numpy() - MIDTOWN_LON) * 84.3,
        (gdf["centroid_lat"].to_numpy() - MIDTOWN_LAT) * 111.0,
    )
    base = gdf["borough"].map(BOROUGH_DEMAND_WEIGHT).fillna(0.01).to_numpy()
    decay = np.exp(-gdf["km_from_midtown"].to_numpy() / 4.5)
    gdf["demand_weight"] = base * (0.15 + 0.85 * decay)
    gdf.loc[gdf["is_airport"], "demand_weight"] = 0.55
    return gdf.sort_values("zone_id").reset_index(drop=True)


def zone_h3_cells(zones: ZoneSet, resolution: int) -> pd.DataFrame:
    import h3

    rows: list[dict] = []
    for zid, geom in zip(zones.gdf["zone_id"], zones.gdf["geometry"], strict=True):
        poly = geom if geom.geom_type == "Polygon" else unary_union(geom)
        polys = [poly] if poly.geom_type == "Polygon" else list(poly.geoms)
        cells: set[str] = set()
        for p in polys:
            coords = [[lat, lon] for lon, lat in p.exterior.coords]
            try:
                cells |= set(h3.polygon_to_cells(h3.LatLngPoly(coords), resolution))
            except Exception:
                continue
        if not cells:
            c = poly.centroid
            cells = {h3.latlng_to_cell(c.y, c.x, resolution)}
        rows.extend({"zone_id": int(zid), "h3_cell": c} for c in cells)

    out = pd.DataFrame(rows)
    log_event(
        log,
        "H3 index built",
        resolution=resolution,
        cells=len(out),
        cells_per_zone=round(len(out) / max(len(zones.gdf), 1), 1),
    )
    return out


def get_zones(cfg, prefer_real: bool = True) -> ZoneSet:
    external = cfg.path("external")
    shp_dir = external / "taxi_zones"
    lookup = external / "taxi_zone_lookup.csv"
    shp = find_shapefile(shp_dir) if shp_dir.exists() else None

    if prefer_real and lookup.exists() and shp is not None:
        return load_tlc_zones(shp_dir, lookup)

    if prefer_real:
        log.warning(
            "FALLING BACK TO SYNTHETIC GEOMETRY. Real taxi zone data was requested but "
            "lookup_present=%s shapefile_present=%s under %s. Maps and spatial features "
            "will not reflect real New York. Run `urbanflow ingest` to fetch it.",
            lookup.exists(),
            shp is not None,
            shp_dir,
        )
    return build_synthetic_zones(seed=int(cfg.project.random_seed))
