from __future__ import annotations

import gc
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from cabcast.logging_utils import get_logger, log_event

log = get_logger(__name__)


@dataclass
class TrainedModel:
    booster: lgb.Booster
    feature_cols: list[str]
    best_iteration: int
    params: dict

    def predict(self, data) -> np.ndarray:
        matrix = (
            data
            if isinstance(data, np.ndarray)
            else data[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
        )
        raw = self.booster.predict(matrix, num_iteration=self.best_iteration or None)
        return np.clip(np.asarray(raw, dtype=float), 0.0, None)

    def importance(self, kind: str = "gain") -> pd.DataFrame:
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_cols,
                    "importance": self.booster.feature_importance(importance_type=kind),
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


def _dataset(
    matrix: np.ndarray,
    labels: np.ndarray,
    feature_cols: list[str],
    params: dict,
    reference: lgb.Dataset | None = None,
) -> lgb.Dataset:
    dataset = lgb.Dataset(
        matrix,
        label=labels,
        feature_name=list(feature_cols),
        params=params,
        reference=reference,
        free_raw_data=True,
    )
    dataset.construct()
    gc.collect()
    return dataset


def _base_params(cfg, seed: int) -> dict:
    m = cfg.model.lgbm
    return {
        "objective": m.objective,
        "tweedie_variance_power": float(m.tweedie_variance_power),
        "learning_rate": float(m.learning_rate),
        "num_leaves": int(m.num_leaves),
        "min_child_samples": int(m.min_child_samples),
        "subsample": float(m.subsample),
        "subsample_freq": int(m.subsample_freq),
        "colsample_bytree": float(m.colsample_bytree),
        "reg_lambda": float(m.reg_lambda),
        "max_bin": int(m.get("max_bin", 255)),
        "metric": "mae",
        "verbosity": -1,
        "num_threads": int(cfg.project.get("num_threads", 0)),
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
    }


def _fit(
    params: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray | None,
    y_valid: np.ndarray | None,
    feature_cols: list[str],
    cfg,
) -> TrainedModel:
    dtrain = _dataset(x_train, y_train, feature_cols, params)
    callbacks = [lgb.log_evaluation(period=0)]
    valid_sets = []
    if x_valid is not None and len(x_valid):
        valid_sets = [_dataset(x_valid, y_valid, feature_cols, params, reference=dtrain)]
        callbacks.append(
            lgb.early_stopping(int(cfg.model.lgbm.early_stopping_rounds), verbose=False)
        )

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=int(cfg.model.lgbm.n_estimators),
        valid_sets=valid_sets,
        callbacks=callbacks,
    )
    best = booster.best_iteration or booster.current_iteration()
    del dtrain, valid_sets
    gc.collect()
    return TrainedModel(booster, list(feature_cols), int(best), params)


def train_point_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray | None,
    y_valid: np.ndarray | None,
    feature_cols: list[str],
    cfg,
) -> TrainedModel:
    params = _base_params(cfg, int(cfg.project.random_seed))
    model = _fit(params, x_train, y_train, x_valid, y_valid, feature_cols, cfg)
    log_event(
        log, "point model trained", rows=len(x_train), features=len(feature_cols),
        best_iteration=model.best_iteration,
    )
    return model


def train_quantile_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray | None,
    y_valid: np.ndarray | None,
    feature_cols: list[str],
    cfg,
    quantile: float,
) -> TrainedModel:
    params = _base_params(cfg, int(cfg.project.random_seed))
    params.update({"objective": "quantile", "alpha": float(quantile), "metric": "quantile"})
    params.pop("tweedie_variance_power", None)
    model = _fit(params, x_train, y_train, x_valid, y_valid, feature_cols, cfg)
    log_event(
        log, "quantile model trained", quantile=quantile, best_iteration=model.best_iteration
    )
    return model
