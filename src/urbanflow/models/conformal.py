from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return float("inf")
    level = min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


@dataclass
class SplitConformal:
    alpha: float = 0.1
    qhat_: float = field(default=float("nan"), init=False)

    def calibrate(self, y: np.ndarray, yhat: np.ndarray) -> SplitConformal:
        self.qhat_ = conformal_quantile(np.abs(y - yhat), self.alpha)
        return self

    def interval(self, yhat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lower = np.clip(yhat - self.qhat_, 0.0, None)
        upper = yhat + self.qhat_
        return lower, upper


@dataclass
class ConformalizedQuantile:
    alpha: float = 0.1
    qhat_: float = field(default=float("nan"), init=False)

    @staticmethod
    def scores(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        return np.maximum(lo - y, y - hi)

    def calibrate(self, y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> ConformalizedQuantile:
        self.qhat_ = conformal_quantile(self.scores(y, lo, hi), self.alpha)
        return self

    def interval(self, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.clip(lo - self.qhat_, 0.0, None), hi + self.qhat_


@dataclass
class MondrianConformalizedQuantile:
    alpha: float = 0.1
    n_bins: int = 6
    edges_: np.ndarray = field(default_factory=lambda: np.array([]), init=False)
    qhat_: dict[int, float] = field(default_factory=dict, init=False)
    global_qhat_: float = field(default=float("nan"), init=False)

    def _assign(self, strat_value: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self.edges_, strat_value, side="right") - 1, 0, len(self.edges_) - 2)

    def calibrate(
        self,
        y: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        strat_value: np.ndarray,
    ) -> MondrianConformalizedQuantile:
        qs = np.linspace(0, 100, self.n_bins + 1)
        self.edges_ = np.unique(np.percentile(strat_value, qs))
        if len(self.edges_) < 3:
            self.edges_ = np.array([strat_value.min(), strat_value.max() + 1e-9])
        self.edges_[0] = -np.inf
        self.edges_[-1] = np.inf

        scores = ConformalizedQuantile.scores(y, lo, hi)
        self.global_qhat_ = conformal_quantile(scores, self.alpha)
        bins = self._assign(strat_value)
        for b in np.unique(bins):
            mask = bins == b
            self.qhat_[int(b)] = (
                conformal_quantile(scores[mask], self.alpha)
                if mask.sum() >= 50
                else self.global_qhat_
            )
        return self

    def interval(
        self, lo: np.ndarray, hi: np.ndarray, strat_value: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        bins = self._assign(strat_value)
        q = np.array([self.qhat_.get(int(b), self.global_qhat_) for b in bins])
        return np.clip(lo - q, 0.0, None), hi + q

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "bin": list(self.qhat_),
                "qhat": [self.qhat_[b] for b in self.qhat_],
            }
        ).sort_values("bin").reset_index(drop=True)
