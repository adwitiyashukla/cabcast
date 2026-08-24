from __future__ import annotations

import pytest

from urbanflow.config import ROOT, apply_dotted_overrides, load_config


def test_config_loads_expected_sections(cfg):
    for section in ("project", "paths", "data", "quality", "geo", "features", "split", "model"):
        assert section in cfg


def test_dotted_override_types():
    data = apply_dotted_overrides(
        {"model": {"lgbm": {"num_leaves": 128}}},
        ["model.lgbm.num_leaves=64", "model.lgbm.learning_rate=0.01", "geo.flag=true"],
    )
    assert data["model"]["lgbm"]["num_leaves"] == 64
    assert data["model"]["lgbm"]["learning_rate"] == pytest.approx(0.01)
    assert data["geo"]["flag"] is True


def test_override_rejects_malformed():
    with pytest.raises(ValueError):
        apply_dotted_overrides({}, ["no_equals_sign"])


def test_paths_resolve_under_repo_root(cfg):
    assert cfg.path("gold").is_absolute()
    assert str(cfg.path("gold")).startswith(str(ROOT))


def test_load_config_applies_overrides():
    cfg = load_config(overrides=["geo.h3_resolution=9"])
    assert cfg.geo.h3_resolution == 9
