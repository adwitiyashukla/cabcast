from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cabcast.config import load_config
from cabcast.logging_utils import get_logger, log_event

log = get_logger("cabcast.cli")

DESCRIPTION = (
    "CabCast: NYC taxi demand forecasting with calibrated uncertainty "
    "and optimal-transport fleet rebalancing"
)


def _context(cfg, prefer_real: bool = True):
    from cabcast.geo.graph import build_zone_graph
    from cabcast.geo.zones import get_zones

    zones = get_zones(cfg, prefer_real=prefer_real)
    graph = build_zone_graph(zones, int(cfg.geo.laplacian_eigenvectors))
    return zones, graph


def cmd_ingest(cfg, args) -> int:
    from cabcast.data.bronze import ingest

    written = ingest(cfg, source=args.source)
    print(f"ingested {len(written)} months, {sum(written.values()):,} rows")
    return 0


def cmd_silver(cfg, args) -> int:
    from cabcast.data.silver import build_silver

    path, report = build_silver(cfg)
    print(f"silver written to {path.name}: {report.rows_out:,} rows kept, "
          f"{report.quarantined:,} quarantined ({report.quarantine_rate:.2%})")
    return 0


def cmd_gold(cfg, args) -> int:
    from cabcast.data.gold import build_gold, load_panel, panel_summary

    zones, graph = _context(cfg)
    silver_path = cfg.path("silver") / "trips.parquet"
    path, _ = build_gold(cfg, silver_path, zones, graph)
    print(json.dumps(panel_summary(load_panel(path)), indent=2))
    return 0


def cmd_train(cfg, args) -> int:
    from cabcast.data.gold import load_panel
    from cabcast.pipelines.train import persist, run_training

    zones, graph = _context(cfg)
    panel = load_panel(cfg.path("gold") / "demand_panel.parquet")
    artifacts = run_training(panel, graph, cfg, data_source=zones.source)
    path = persist(artifacts, cfg, zones.source)
    test = artifacts.results["test"]["point"]["lightgbm"]
    print(f"training complete, results at {path.name}")
    print(f"test MAE {test['mae']:.4f}  RMSE {test['rmse']:.4f}  MASE {test.get('mase', float('nan')):.4f}")
    return 0


REPORT_PANEL_COLUMNS = ["zone_id", "hour_ts", "trips", "in_crz", "borough"]


def cmd_report(cfg, args) -> int:
    import gc

    from cabcast.data.gold import load_panel
    from cabcast.pipelines.report import generate
    from cabcast.pipelines.train import load_artifacts, persist, run_training

    zones, graph = _context(cfg)
    gold_path = cfg.path("gold") / "demand_panel.parquet"

    if args.reuse:
        artifacts = load_artifacts(cfg)
    else:
        panel = load_panel(gold_path)
        artifacts = run_training(panel, graph, cfg, data_source=zones.source)
        persist(artifacts, cfg, zones.source)
        del panel
        gc.collect()

    slim = load_panel(gold_path, REPORT_PANEL_COLUMNS)
    results = generate(artifacts, slim, zones, graph, cfg)
    from cabcast.pipelines.summary import write_summary

    write_summary(results, cfg)
    print(f"generated {len(results['figures'])} figures in {cfg.path('figures')}")
    return 0


def cmd_all(cfg, args) -> int:
    import gc

    from cabcast.data.bronze import ingest
    from cabcast.data.gold import build_gold, load_panel
    from cabcast.data.silver import build_silver
    from cabcast.pipelines.report import generate
    from cabcast.pipelines.train import persist, run_training

    ingest(cfg, source=args.source)
    zones, graph = _context(cfg)
    silver_path, _ = build_silver(cfg)
    gold_path, _ = build_gold(cfg, silver_path, zones, graph)

    panel = load_panel(gold_path)
    artifacts = run_training(panel, graph, cfg, data_source=zones.source)
    persist(artifacts, cfg, zones.source)
    del panel
    gc.collect()

    slim = load_panel(gold_path, REPORT_PANEL_COLUMNS)
    results = generate(artifacts, slim, zones, graph, cfg)

    from cabcast.pipelines.summary import write_summary

    write_summary(results, cfg)
    print("\npipeline complete")
    print(f"  figures  {cfg.path('figures')}")
    print(f"  results  {cfg.path('reports') / 'results.json'}")
    print(f"  models   {cfg.path('artifacts')}")
    return 0


def cmd_serve(cfg, args) -> int:
    import uvicorn

    uvicorn.run(
        "cabcast.serving.api:app",
        host=str(cfg.serving.host),
        port=int(args.port or cfg.serving.port),
        reload=False,
    )
    return 0


COMMANDS = {
    "ingest": cmd_ingest,
    "silver": cmd_silver,
    "gold": cmd_gold,
    "train": cmd_train,
    "report": cmd_report,
    "all": cmd_all,
    "serve": cmd_serve,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cabcast", description=DESCRIPTION)
    parser.add_argument("command", choices=sorted(COMMANDS), help="pipeline stage to run")
    parser.add_argument("--config", type=Path, default=None, help="path to a config YAML")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        help="config override as key.subkey=value, repeatable",
    )
    parser.add_argument(
        "--source", choices=["auto", "remote", "synthetic"], default="auto",
        help="where trip data comes from",
    )
    parser.add_argument("--port", type=int, default=None, help="port for the serve command")
    parser.add_argument(
        "--reuse", action="store_true",
        help="report from saved artifacts instead of retraining",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    log_event(log, "cabcast start", command=args.command, overrides=args.overrides)
    return COMMANDS[args.command](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
