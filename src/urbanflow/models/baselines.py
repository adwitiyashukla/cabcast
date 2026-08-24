from __future__ import annotations

import numpy as np
import pandas as pd


class SeasonalNaive:
    def __init__(self, season_hours: int = 168) -> None:
        self.season_hours = season_hours
        self.column = f"lag_{season_hours}h"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.column not in df.columns:
            raise KeyError(f"{self.column} required for SeasonalNaive")
        return np.nan_to_num(df[self.column].to_numpy(dtype=float), nan=0.0)


class HistoricalMean:
    def __init__(self, keys: tuple[str, ...] = ("zone_id", "hour_of_week")) -> None:
        self.keys = list(keys)
        self.table_: pd.DataFrame | None = None
        self.global_mean_: float = 0.0

    def fit(self, df: pd.DataFrame, target: str = "trips") -> HistoricalMean:
        self.global_mean_ = float(df[target].mean())
        self.table_ = (
            df.groupby(self.keys, observed=True)[target]
            .mean()
            .rename("prediction")
            .reset_index()
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.table_ is None:
            raise RuntimeError("HistoricalMean must be fitted first")
        merged = df[self.keys].merge(self.table_, on=self.keys, how="left")
        return merged["prediction"].fillna(self.global_mean_).to_numpy(dtype=float)


class DriftedSeasonalNaive:
    def __init__(self, season_hours: int = 168, recent_hours: int = 24) -> None:
        self.season = f"lag_{season_hours}h"
        self.recent = f"roll_mean_{recent_hours}h"
        self.long = "roll_mean_168h"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        base = np.nan_to_num(df[self.season].to_numpy(dtype=float), nan=0.0)
        recent = np.nan_to_num(df[self.recent].to_numpy(dtype=float), nan=0.0)
        long = np.nan_to_num(df[self.long].to_numpy(dtype=float), nan=0.0)
        ratio = np.where(long > 0.5, recent / np.maximum(long, 1e-6), 1.0)
        return np.clip(base * np.clip(ratio, 0.5, 2.0), 0.0, None)
