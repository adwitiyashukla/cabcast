from __future__ import annotations

import hashlib
import io
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import requests

from cabcast.data.synth import month_bounds, month_range, write_synthetic_bronze
from cabcast.logging_utils import get_logger, log_event, stage

log = get_logger(__name__)

TIMEOUT = 120
CHUNK = 1 << 20
RETRIES = 3

WEATHER_FIELDS = (
    "temperature_2m,precipitation,snowfall,wind_speed_10m,relative_humidity_2m,cloud_cover"
)


def _row_count(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def _sha256(path: Path, limit: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(limit))
    return h.hexdigest()[:16]


def _download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return False
    tmp = dest.with_suffix(dest.suffix + ".part")
    last: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=TIMEOUT) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(CHUNK):
                        fh.write(chunk)
            tmp.rename(dest)
            return True
        except Exception as exc:
            last = exc
            tmp.unlink(missing_ok=True)
            log_event(log, "download retry", url=url, attempt=attempt, error=repr(exc))
    raise RuntimeError(f"failed to download {url}") from last


def fetch_zone_reference(cfg) -> tuple[Path, Path]:
    external = cfg.path("external")
    lookup = external / "taxi_zone_lookup.csv"
    shp_dir = external / "taxi_zones"

    _download(f"{cfg.data.misc_base_url}/taxi_zone_lookup.csv", lookup)
    from cabcast.geo.zones import find_shapefile

    if not (shp_dir.exists() and find_shapefile(shp_dir) is not None):
        resp = requests.get(f"{cfg.data.misc_base_url}/taxi_zones.zip", timeout=TIMEOUT)
        resp.raise_for_status()
        shp_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            zf.extractall(shp_dir)
    found = find_shapefile(shp_dir)
    log_event(
        log, "zone reference ready", lookup=lookup.name,
        shapefile=str(found.relative_to(shp_dir)) if found else "MISSING",
    )
    return shp_dir, lookup


def fetch_weather(cfg, months: list[tuple[int, int]]) -> Path:
    external = cfg.path("external")
    out = external / "weather_hourly.parquet"
    if out.exists():
        return out

    start = month_bounds(*months[0])[0] - pd.Timedelta(days=31)
    end = month_bounds(*months[-1])[1]
    params = {
        "latitude": cfg.data.weather_lat,
        "longitude": cfg.data.weather_lon,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": WEATHER_FIELDS,
        "timezone": "America/New_York",
    }
    resp = requests.get(cfg.data.weather_url, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()["hourly"]

    df = pd.DataFrame(payload).rename(columns={"time": "hour_ts"})
    df["hour_ts"] = pd.to_datetime(df["hour_ts"])
    for col in df.columns:
        if col != "hour_ts":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.ffill().bfill()
    df.to_parquet(out, index=False)
    log_event(log, "weather downloaded", rows=len(df), start=params["start_date"], end=params["end_date"])
    return out


def adopt_manual_downloads(cfg) -> int:
    drop = Path(cfg.data.manual_drop_dir)
    if not drop.is_absolute():
        from cabcast.config import ROOT

        drop = ROOT / drop
    if not drop.exists():
        return 0

    bronze = cfg.path("bronze")
    external = cfg.path("external")
    adopted = 0

    for src in sorted(drop.glob("*.parquet")):
        dest = bronze / src.name
        if not dest.exists():
            shutil.copy2(src, dest)
            adopted += 1
    for name in ("taxi_zone_lookup.csv",):
        src = drop / name
        if src.exists() and not (external / name).exists():
            shutil.copy2(src, external / name)
            adopted += 1
    zip_src = drop / "taxi_zones.zip"
    if zip_src.exists() and not (external / "taxi_zones").exists():
        with zipfile.ZipFile(zip_src) as zf:
            zf.extractall(external / "taxi_zones")
        adopted += 1
    csv_src = drop / "weather_nyc_hourly.csv"
    out = external / "weather_hourly.parquet"
    if csv_src.exists() and not out.exists():
        raw = pd.read_csv(csv_src, skiprows=2)
        raw = raw.rename(columns={raw.columns[0]: "hour_ts"})
        raw.columns = [c.split(" (")[0] for c in raw.columns]
        raw["hour_ts"] = pd.to_datetime(raw["hour_ts"])
        raw.to_parquet(out, index=False)
        adopted += 1

    if adopted:
        log_event(log, "adopted manual downloads", files=adopted, source=str(drop))
    return adopted


def fetch_remote(cfg, months: list[tuple[int, int]]) -> dict[str, int]:
    bronze = cfg.path("bronze")
    fetch_zone_reference(cfg)
    fetch_weather(cfg, months)

    written: dict[str, int] = {}
    for year, month in months:
        tag = f"{year}-{month:02d}"
        name = f"{cfg.data.service}_tripdata_{tag}.parquet"
        dest = bronze / name
        _download(f"{cfg.data.tlc_base_url}/{name}", dest)
        written[tag] = _row_count(dest)
        log_event(log, "remote month ready", month=tag, rows=written[tag])
    return written


def write_manifest(cfg, written: dict[str, int], source: str) -> Path:
    bronze = cfg.path("bronze")
    rows = []
    for tag, n in sorted(written.items()):
        path = bronze / f"{cfg.data.service}_tripdata_{tag}.parquet"
        rows.append(
            {
                "month": tag,
                "rows": n,
                "bytes": path.stat().st_size if path.exists() else 0,
                "sha256_head": _sha256(path) if path.exists() else "",
                "source": source,
            }
        )
    out = cfg.path("reports") / "bronze_manifest.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def reference_data_present(cfg) -> bool:
    from cabcast.geo.zones import find_shapefile

    external = cfg.path("external")
    return (
        (external / "taxi_zone_lookup.csv").exists()
        and find_shapefile(external / "taxi_zones") is not None
        and (external / "weather_hourly.parquet").exists()
    )


def ingest(cfg, source: str = "auto", months: list[tuple[int, int]] | None = None) -> dict[str, int]:
    months = months or month_range(cfg.data.start_month, cfg.data.end_month)
    bronze = cfg.path("bronze")

    with stage(log, "ingest", source=source, months=len(months)) as st:
        adopt_manual_downloads(cfg)
        existing = {
            f.stem.split("_")[-1]: _row_count(f)
            for f in sorted(bronze.glob(f"{cfg.data.service}_tripdata_*.parquet"))
        }
        wanted = {f"{y}-{m:02d}" for y, m in months}
        have = wanted & set(existing)

        if source == "synthetic" or (source == "auto" and not have and not _remote_reachable(cfg)):
            from cabcast.geo.graph import build_zone_graph
            from cabcast.geo.zones import get_zones

            zones = get_zones(cfg, prefer_real=True)
            graph = build_zone_graph(zones, int(cfg.geo.laplacian_eigenvectors))
            written = write_synthetic_bronze(cfg, zones, graph, months)
            resolved = "synthetic"
        elif have == wanted:
            if not reference_data_present(cfg) and _remote_reachable(cfg):
                fetch_zone_reference(cfg)
                fetch_weather(cfg, months)
            written = {k: existing[k] for k in sorted(have)}
            resolved = "cached"
        else:
            written = fetch_remote(cfg, months)
            resolved = "remote"

        write_manifest(cfg, written, resolved)
        st["resolved_source"] = resolved
        st["rows"] = sum(written.values())
    return written


def _remote_reachable(cfg) -> bool:
    try:
        resp = requests.head(f"{cfg.data.misc_base_url}/taxi_zone_lookup.csv", timeout=8)
        return resp.status_code < 400
    except Exception:
        return False
