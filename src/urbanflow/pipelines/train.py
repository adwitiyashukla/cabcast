from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from urbanflow.evaluation.backtest import (
    Fold,
    assert_temporal_integrity,
    holdout_bounds,
    rolling_origin_folds,
)
from urbanflow.evaluation.metrics import (
    coverage_by_stratum,
    interval_metrics,
    pinball_loss,
    point_metrics,
)
from urbanflow.features.build import assert_no_leakage, build_features
from urbanflow.geo.graph import ZoneGraph
from urbanflow.logging_utils import get_logger, log_event, stage
from urbanflow.models.baselines import DriftedSeasonalNaive, HistoricalMean, SeasonalNaive
from urbanflow.models.conformal import (
    ConformalizedQuantile,
    MondrianConformalizedQuantile,
    SplitConformal,
)
from urbanflow.models.lgbm import TrainedModel, train_point_model, train_quantile_model
from urbanflow.models.registry import ModelRegistry, make_card

log = get_logger(__name__)

TARGET = "trips"
META_COLUMNS = [
    "zone_id",
    "hour_ts",
    "trips",
    "hour_of_week",
    "lag_168h",
    "roll_mean_24h",
    "roll_mean_168h",
]
DRIFT_SAMPLE_ROWS = 60_000


@dataclass
class Matrix:
    x: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame
    hours: np.ndarray
    feature_cols: list[str]

    def frame(self, window: slice) -> pd.DataFrame:
        return self.meta.iloc[window]


@dataclass
class TrainingArtifacts:
    point_model: TrainedModel
    lower_model: TrainedModel
    upper_model: TrainedModel
    cqr: ConformalizedQuantile
    mondrian: MondrianConformalizedQuantile
    split_conformal: SplitConformal
    feature_cols: list[str]
    results: dict[str, Any]
    test_x: np.ndarray
    test_meta: pd.DataFrame
    drift_reference: pd.DataFrame
    calibration: dict[str, np.ndarray]


def to_matrix(features: pd.DataFrame, feature_cols: list[str]) -> Matrix:
    x = np.ascontiguousarray(features[feature_cols].to_numpy(dtype=np.float32))
    y = features[TARGET].to_numpy(dtype=np.float32)
    meta = features[[c for c in META_COLUMNS if c in features.columns]].copy()
    hours = features["hour_ts"].to_numpy()
    return Matrix(x=x, y=y, meta=meta, hours=hours, feature_cols=list(feature_cols))


def _baseline_predictions(
    matrix: Matrix, train_window: slice, eval_window: slice
) -> dict[str, np.ndarray]:
    ev = matrix.frame(eval_window)
    return {
        "seasonal_naive": SeasonalNaive(168).predict(ev),
        "drifted_seasonal_naive": DriftedSeasonalNaive().predict(ev),
        "historical_mean": HistoricalMean().fit(matrix.frame(train_window)).predict(ev),
    }


def _naive_reference(meta: pd.DataFrame) -> np.ndarray:
    return np.nan_to_num(meta["lag_168h"].to_numpy(dtype=float), nan=0.0)


def _backtest(matrix: Matrix, folds: list[Fold], cfg) -> dict[str, Any]:
    per_fold: list[dict[str, Any]] = []

    for fold in folds:
        y = matrix.y[fold.valid].astype(float)
        naive = _naive_reference(matrix.frame(fold.valid))
        preds = _baseline_predictions(matrix, fold.train, fold.valid)

        model = train_point_model(
            matrix.x[fold.train], matrix.y[fold.train],
            matrix.x[fold.valid], matrix.y[fold.valid],
            matrix.feature_cols, cfg,
        )
        preds["lightgbm"] = model.predict(matrix.x[fold.valid])

        row: dict[str, Any] = dict(fold.describe())
        for name, yhat in preds.items():
            for metric, value in point_metrics(y, yhat, naive).items():
                row[f"{name}.{metric}"] = round(value, 6)
        per_fold.append(row)
        log_event(
            log, "fold complete", fold=fold.name,
            lightgbm_mae=row["lightgbm.mae"], seasonal_naive_mae=row["seasonal_naive.mae"],
        )
        del model, preds
        gc.collect()

    table = pd.DataFrame(per_fold)
    metric_cols = [c for c in table.columns if "." in c]
    return {"per_fold": per_fold, "mean": table[metric_cols].mean().round(6).to_dict()}


def _final_windows(matrix: Matrix, train: slice, cfg) -> tuple[slice, slice, slice]:
    hours = matrix.hours[train]
    cal_cutoff = pd.Timestamp(hours[-1]) - pd.Timedelta(days=int(cfg.conformal.calibration_days))
    cal_start = int(np.searchsorted(hours, np.datetime64(cal_cutoff), side="right"))

    val_cutoff = cal_cutoff - pd.Timedelta(days=int(cfg.split.validation_days))
    val_start = int(np.searchsorted(hours, np.datetime64(val_cutoff), side="right"))

    fit_window = slice(train.start, train.start + val_start)
    val_window = slice(train.start + val_start, train.start + cal_start)
    cal_window = slice(train.start + cal_start, train.stop)
    log_event(
        log, "final fit split",
        n_fit=fit_window.stop - fit_window.start,
        n_val=val_window.stop - val_window.start,
        n_calibration=cal_window.stop - cal_window.start,
    )
    return fit_window, val_window, cal_window


def _evaluate_test(
    matrix: Matrix,
    train: slice,
    test: slice,
    models: tuple[TrainedModel, TrainedModel, TrainedModel],
    cqr: ConformalizedQuantile,
    mondrian: MondrianConformalizedQuantile,
    split_conf: SplitConformal,
    cfg,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    point, lower, upper = models
    y = matrix.y[test].astype(float)
    test_meta = matrix.frame(test)
    naive = _naive_reference(test_meta)
    alpha = float(cfg.conformal.alpha)

    preds = _baseline_predictions(matrix, train, test)
    preds["lightgbm"] = point.predict(matrix.x[test])
    point_table = {
        name: {k: round(v, 6) for k, v in point_metrics(y, yhat, naive).items()}
        for name, yhat in preds.items()
    }

    raw_lo = lower.predict(matrix.x[test])
    raw_hi = np.maximum(upper.predict(matrix.x[test]), raw_lo)
    yhat = preds["lightgbm"]

    intervals: dict[str, Any] = {
        "quantile_uncalibrated": {
            **interval_metrics(y, raw_lo, raw_hi, alpha),
            "method": "LightGBM quantile regression, no conformal correction",
        }
    }
    lo_c, hi_c = cqr.interval(raw_lo, raw_hi)
    intervals["conformalized_quantile"] = {
        **interval_metrics(y, lo_c, hi_c, alpha),
        "qhat": round(cqr.qhat_, 5),
        "method": "CQR, single global correction",
    }
    lo_m, hi_m = mondrian.interval(raw_lo, raw_hi, yhat)
    intervals["mondrian_cqr"] = {
        **interval_metrics(y, lo_m, hi_m, alpha),
        "method": "Mondrian CQR, correction conditioned on predicted demand stratum",
    }
    lo_s, hi_s = split_conf.interval(yhat)
    intervals["split_conformal"] = {
        **interval_metrics(y, lo_s, hi_s, alpha),
        "qhat": round(split_conf.qhat_, 5),
        "method": "Split conformal on absolute residuals",
    }

    strata = pd.qcut(yhat, q=5, labels=False, duplicates="drop")
    conditional = coverage_by_stratum(y, lo_c, hi_c, strata).merge(
        coverage_by_stratum(y, lo_m, hi_m, strata)[["stratum", "coverage", "mean_width"]],
        on="stratum", suffixes=("_cqr", "_mondrian"),
    )

    pinball = {
        f"q{int(q * 100):02d}": round(pinball_loss(y, raw_lo if q <= 0.5 else raw_hi, q), 6)
        for q in (min(cfg.model.quantiles), max(cfg.model.quantiles))
    }

    predictions = {
        "zone_id": test_meta["zone_id"].to_numpy(),
        "hour_ts": test_meta["hour_ts"].to_numpy(),
        "y": y,
        "yhat": yhat,
        "q_lower": raw_lo,
        "q_upper": raw_hi,
        "lower": lo_m,
        "upper": hi_m,
    }
    evaluation = {
        "point": point_table,
        "intervals": intervals,
        "conditional_coverage": conditional.round(5).to_dict(orient="records"),
        "pinball": pinball,
        "n_test_rows": int(len(y)),
    }
    return evaluation, predictions


def run_training(panel: pd.DataFrame, graph: ZoneGraph, cfg, data_source: str) -> TrainingArtifacts:
    with stage(log, "training_pipeline", rows=len(panel)):
        features, feature_cols = build_features(panel, graph, cfg)
        assert_no_leakage(feature_cols)
        del panel
        gc.collect()

        matrix = to_matrix(features, feature_cols)
        date_range = [str(features["hour_ts"].min()), str(features["hour_ts"].max())]
        n_rows, n_zones = len(features), int(features["zone_id"].nunique())

        rng = np.random.default_rng(int(cfg.project.random_seed))
        del features
        gc.collect()

        train, test, _ = holdout_bounds(matrix.hours, int(cfg.split.test_days))
        folds = rolling_origin_folds(
            matrix.hours[train], int(cfg.split.n_folds),
            int(cfg.split.validation_days), int(cfg.split.embargo_hours),
        )
        assert_temporal_integrity(matrix.hours[train], folds, int(cfg.split.embargo_hours))

        backtest = _backtest(matrix, folds, cfg)

        fit_w, val_w, cal_w = _final_windows(matrix, train, cfg)
        quantiles = list(cfg.model.quantiles)
        point = train_point_model(
            matrix.x[fit_w], matrix.y[fit_w], matrix.x[val_w], matrix.y[val_w],
            feature_cols, cfg,
        )
        lower = train_quantile_model(
            matrix.x[fit_w], matrix.y[fit_w], matrix.x[val_w], matrix.y[val_w],
            feature_cols, cfg, min(quantiles),
        )
        upper = train_quantile_model(
            matrix.x[fit_w], matrix.y[fit_w], matrix.x[val_w], matrix.y[val_w],
            feature_cols, cfg, max(quantiles),
        )

        alpha = float(cfg.conformal.alpha)
        y_cal = matrix.y[cal_w].astype(float)
        cal_lo = lower.predict(matrix.x[cal_w])
        cal_hi = np.maximum(upper.predict(matrix.x[cal_w]), cal_lo)
        cal_point = point.predict(matrix.x[cal_w])

        cqr = ConformalizedQuantile(alpha).calibrate(y_cal, cal_lo, cal_hi)
        mondrian = MondrianConformalizedQuantile(
            alpha, int(cfg.conformal.mondrian_bins)
        ).calibrate(y_cal, cal_lo, cal_hi, cal_point)
        split_conf = SplitConformal(alpha).calibrate(y_cal, cal_point)

        test_eval, predictions = _evaluate_test(
            matrix, train, test, (point, lower, upper), cqr, mondrian, split_conf, cfg
        )

        gains = point.importance("gain")
        gains["importance_pct"] = (gains["importance"] / gains["importance"].sum() * 100.0).round(3)

        n_train = train.stop - train.start
        take = rng.choice(n_train, size=min(DRIFT_SAMPLE_ROWS, n_train), replace=False)
        drift_reference = pd.DataFrame(matrix.x[train][take], columns=feature_cols)

        test_x = np.ascontiguousarray(matrix.x[test])
        test_meta = matrix.frame(test).reset_index(drop=True)

        results = {
            "data_source": data_source,
            "n_rows_features": n_rows,
            "n_features": len(feature_cols),
            "n_zones": n_zones,
            "date_range": date_range,
            "backtest": backtest,
            "test": test_eval,
            "conformal": {
                "alpha": alpha,
                "calibration_rows": int(cal_w.stop - cal_w.start),
                "cqr_qhat": round(cqr.qhat_, 5),
                "split_conformal_qhat": round(split_conf.qhat_, 5),
                "mondrian_bins": mondrian.summary().to_dict(orient="records"),
            },
            "feature_importance": gains.head(25).to_dict(orient="records"),
            "predictions": dict(predictions),
        }
        calibration = {"y": y_cal, "lower": cal_lo, "upper": cal_hi, "point": cal_point}
        del matrix
        gc.collect()

    return TrainingArtifacts(
        point_model=point, lower_model=lower, upper_model=upper,
        cqr=cqr, mondrian=mondrian, split_conformal=split_conf,
        feature_cols=feature_cols, results=results,
        test_x=test_x, test_meta=test_meta,
        drift_reference=drift_reference, calibration=calibration,
    )


def persist(artifacts: TrainingArtifacts, cfg, data_source: str) -> Path:
    registry = ModelRegistry(cfg.path("artifacts"))
    n_rows = artifacts.results["n_rows_features"]
    test_mae = artifacts.results["test"]["point"]["lightgbm"]["mae"]

    for name, model, metrics in (
        ("demand_point", artifacts.point_model, {"test_mae": test_mae}),
        ("demand_q_lower", artifacts.lower_model, {}),
        ("demand_q_upper", artifacts.upper_model, {}),
    ):
        registry.save(name, model, make_card(name, model, n_rows, data_source, metrics))

    (cfg.path("artifacts") / "conformal.json").write_text(
        json.dumps(
            {
                "alpha": artifacts.cqr.alpha,
                "cqr_qhat": artifacts.cqr.qhat_,
                "split_conformal_qhat": artifacts.split_conformal.qhat_,
                "mondrian_edges": artifacts.mondrian.edges_.tolist(),
                "mondrian_qhat": {str(k): v for k, v in artifacts.mondrian.qhat_.items()},
                "mondrian_global_qhat": artifacts.mondrian.global_qhat_,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    store = pd.DataFrame(artifacts.test_x, columns=artifacts.feature_cols)
    store.insert(0, "hour_ts", artifacts.test_meta["hour_ts"].to_numpy())
    store.insert(0, "zone_id", artifacts.test_meta["zone_id"].to_numpy())
    store.to_parquet(cfg.path("artifacts") / "feature_store.parquet", index=False)

    artifacts.drift_reference.to_parquet(
        cfg.path("artifacts") / "drift_reference.parquet", index=False
    )
    pd.DataFrame(artifacts.calibration).to_parquet(
        cfg.path("artifacts") / "calibration.parquet", index=False
    )
    pd.DataFrame(artifacts.results["predictions"]).to_parquet(
        cfg.path("artifacts") / "test_predictions.parquet", index=False
    )
    (cfg.path("artifacts") / "meta.json").write_text(
        json.dumps(
            {
                "feature_cols": artifacts.feature_cols,
                "test_meta_columns": list(artifacts.test_meta.columns),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    payload = {k: v for k, v in artifacts.results.items() if k != "predictions"}
    results_path = cfg.path("reports") / "results.json"
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log_event(log, "artifacts persisted", registry=str(cfg.path("artifacts")))
    return results_path


def load_artifacts(cfg) -> TrainingArtifacts:
    registry = ModelRegistry(cfg.path("artifacts"))
    root = cfg.path("artifacts")
    point, _, _ = registry.load("demand_point")
    lower, _, _ = registry.load("demand_q_lower")
    upper, _, _ = registry.load("demand_q_upper")

    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    feature_cols = meta["feature_cols"]
    payload = json.loads((root / "conformal.json").read_text(encoding="utf-8"))
    alpha = float(payload["alpha"])

    cqr = ConformalizedQuantile(alpha)
    cqr.qhat_ = float(payload["cqr_qhat"])
    split_conf = SplitConformal(alpha)
    split_conf.qhat_ = float(payload["split_conformal_qhat"])
    mondrian = MondrianConformalizedQuantile(alpha, len(payload["mondrian_qhat"]))
    mondrian.edges_ = np.array(payload["mondrian_edges"], dtype=float)
    mondrian.qhat_ = {int(k): float(v) for k, v in payload["mondrian_qhat"].items()}
    mondrian.global_qhat_ = float(payload["mondrian_global_qhat"])

    store = pd.read_parquet(root / "feature_store.parquet")
    predictions = pd.read_parquet(root / "test_predictions.parquet")
    calibration = pd.read_parquet(root / "calibration.parquet")
    results = json.loads((cfg.path("reports") / "results.json").read_text(encoding="utf-8"))
    results["predictions"] = {c: predictions[c].to_numpy() for c in predictions.columns}

    log_event(log, "artifacts reloaded", registry=str(root), rows=len(store))
    return TrainingArtifacts(
        point_model=point, lower_model=lower, upper_model=upper,
        cqr=cqr, mondrian=mondrian, split_conformal=split_conf,
        feature_cols=feature_cols, results=results,
        test_x=np.ascontiguousarray(store[feature_cols].to_numpy(dtype=np.float32)),
        test_meta=store[["zone_id", "hour_ts"]].copy(),
        drift_reference=pd.read_parquet(root / "drift_reference.parquet"),
        calibration={c: calibration[c].to_numpy() for c in calibration.columns},
    )
