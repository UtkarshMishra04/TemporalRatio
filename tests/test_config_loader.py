from __future__ import annotations

from pathlib import Path

import pytest

from temporal_ratio.config import loader
from temporal_ratio.config.schema import ConfigError

ROOT = Path(__file__).resolve().parents[1]
INFER = sorted((ROOT / "configs/infer").rglob("*.yaml"))


@pytest.mark.parametrize("path", INFER, ids=lambda p: str(p.relative_to(ROOT)))
def test_inference_presets_validate(path):
    assert loader.load_inference(path).policy_config


def test_overrides_are_typed():
    exp = loader.load_inference(
        ROOT / "configs/infer/libero/tr_adaptive.yaml",
        ["action_denoising_steps=5", "guidance.w_lang=1", "tr_log_layers=[13]"],
    )
    assert exp.action_denoising_steps == 5
    assert exp.guidance.w_lang == 1.0 and isinstance(exp.guidance.w_lang, float)
    assert exp.tr_log_layers == [13]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("guidance.w_langz=1", "unknown keys"),
        ("guidance.type=cfg", "not one of"),
        ("action_denoising_steps=abc", "expected an integer"),
        ("action_denoising_steps=0", "action_denoising_steps must be positive"),
        ("tr_log_layers.9=1", "out of range"),
    ],
)
def test_invalid_configs_are_rejected(override, message):
    with pytest.raises(ConfigError, match=message):
        loader.load_inference(ROOT / "configs/infer/libero/tr_adaptive.yaml", [override])


def test_env_expansion(monkeypatch, tmp_path):
    cfg = tmp_path / "x.yaml"
    cfg.write_text("checkpoint: ${MY_CKPT:-fallback/dir}\npolicy_config: videoVLA_libero_short\n")
    monkeypatch.delenv("MY_CKPT", raising=False)
    assert loader.load_inference(cfg).checkpoint == "fallback/dir"
    monkeypatch.setenv("MY_CKPT", "/data/ckpt/100")
    assert loader.load_inference(cfg).checkpoint == "/data/ckpt/100"
    cfg.write_text("checkpoint: ${NOT_SET_ANYWHERE}\npolicy_config: x\n")
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
    with pytest.raises(ConfigError, match="not set"):
        loader.load_inference(cfg)


def test_circular_base_is_detected(tmp_path):
    (tmp_path / "a.yaml").write_text("_base_: b.yaml\n")
    (tmp_path / "b.yaml").write_text("_base_: a.yaml\n")
    with pytest.raises(ConfigError, match="circular"):
        loader.load_yaml_tree(tmp_path / "a.yaml")


def test_inference_validation():
    path = ROOT / "configs/infer/libero/tr_adaptive.yaml"
    with pytest.raises(ConfigError, match="velocity space"):
        loader.load_inference(path, ["guidance.space=action"])
    with pytest.raises(ConfigError, match="feature_extraction_step"):
        loader.load_inference(path, ["feature_extraction_step=9"])
    with pytest.raises(ConfigError, match="tr_baseline_fixed"):
        loader.load_inference(path, ["guidance.tr_baseline=fixed"])
