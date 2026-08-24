from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from urbanflow.logging_utils import get_logger, log_event
from urbanflow.models.lgbm import TrainedModel

log = get_logger(__name__)


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


@dataclass
class ModelCard:
    name: str
    stage: str
    created_utc: str
    git_sha: str
    python_version: str
    lightgbm_version: str
    n_features: int
    n_train_rows: int
    best_iteration: int
    data_source: str
    target: str
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelRegistry:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, name: str) -> Path:
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(
        self,
        name: str,
        model: TrainedModel,
        card: ModelCard,
        extras: dict[str, Any] | None = None,
    ) -> Path:
        d = self._dir(name)
        model.booster.save_model(str(d / "model.txt"), num_iteration=model.best_iteration or None)
        (d / "features.json").write_text(json.dumps(model.feature_cols, indent=2), encoding="utf-8")
        (d / "card.json").write_text(json.dumps(card.to_dict(), indent=2), encoding="utf-8")
        if extras:
            (d / "extras.json").write_text(
                json.dumps(extras, indent=2, default=_json_default), encoding="utf-8"
            )
        log_event(log, "model saved", name=name, path=str(d), best_iteration=model.best_iteration)
        return d

    def load(self, name: str) -> tuple[TrainedModel, ModelCard, dict[str, Any]]:
        d = self.root / name
        booster = lgb.Booster(model_file=str(d / "model.txt"))
        features = json.loads((d / "features.json").read_text(encoding="utf-8"))
        card = ModelCard(**json.loads((d / "card.json").read_text(encoding="utf-8")))
        extras_path = d / "extras.json"
        extras = json.loads(extras_path.read_text(encoding="utf-8")) if extras_path.exists() else {}
        model = TrainedModel(booster, features, booster.best_iteration or 0, {})
        return model, card, extras

    def list_models(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if (p / "model.txt").exists())


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def make_card(
    name: str,
    model: TrainedModel,
    n_train_rows: int,
    data_source: str,
    metrics: dict[str, float] | None = None,
    stage: str = "production",
    target: str = "trips",
) -> ModelCard:
    return ModelCard(
        name=name,
        stage=stage,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_sha=git_sha(),
        python_version=platform.python_version(),
        lightgbm_version=lgb.__version__,
        n_features=len(model.feature_cols),
        n_train_rows=int(n_train_rows),
        best_iteration=int(model.best_iteration),
        data_source=data_source,
        target=target,
        metrics=metrics or {},
        params={k: v for k, v in model.params.items() if k != "seed"},
    )
