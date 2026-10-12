from __future__ import annotations

from pathlib import Path

import pytest

from temporal_ratio.config import loader
from temporal_ratio.config.schema import DEFAULT_LORA_TARGETS, ConfigError

ROOT = Path(__file__).resolve().parents[1]
TRAIN = sorted(p for p in (ROOT / "configs/train").rglob("*.yaml") if not p.name.startswith("_"))
INFER = sorted((ROOT / "configs/infer").rglob("*.yaml"))
EVAL = sorted((ROOT / "configs/eval").rglob("*.yaml"))


@pytest.mark.parametrize("path", TRAIN, ids=lambda p: str(p.relative_to(ROOT)))
def test_train_presets_validate(path):
    exp = loader.load_train(path)
    assert exp.name
    assert exp.stages


@pytest.mark.parametrize("path", INFER, ids=lambda p: str(p.relative_to(ROOT)))
def test_inference_presets_validate(path):
    assert loader.load_inference(path).policy_config


@pytest.mark.parametrize("path", EVAL, ids=lambda p: str(p.relative_to(ROOT)))
def test_eval_presets_validate(path):
    exp = loader.load_libero_eval(path)
    assert exp.inference.guidance.type == "tr_adaptive"


def test_base_inheritance_and_lora_override():
    exp = loader.load_train(ROOT / "configs/train/libero/joint_lora_r64.yaml")
    assert exp.adaptation.lora.rank == 64 and exp.adaptation.lora.alpha == 64
    assert exp.adaptation.lora.target_modules == list(DEFAULT_LORA_TARGETS)
    assert exp.data.action_dataset == "videoVLA_libero_short"


def test_overrides_are_typed():
    exp = loader.load_train(
        ROOT / "configs/train/libero/joint_lora_r32.yaml",
        ["stages.0.steps=200", "adaptation.lora.rank=8", "adaptation.lora.target_modules=[q_proj, v_proj]"],
    )
    assert exp.stages[0].steps == 200
    assert exp.adaptation.lora.rank == 8
    assert exp.adaptation.lora.target_modules == ["q_proj", "v_proj"]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("stages.0.stepz=10", "unknown keys"),
        ("adaptation.video=qlora", "not one of"),
        ("adaptation.lora.target_modules=[attn]", "invalid LoRA target_modules"),
        ("stages.0.mode=video_only", "requires stage modes"),
        ("stages.0.steps=-5", "steps must be positive"),
        ("adaptation.lora.rank=abc", "expected an integer"),
        ("stages.9.steps=1", "out of range"),
    ],
)
def test_invalid_configs_are_rejected(override, message):
    with pytest.raises(ConfigError, match=message):
        loader.load_train(ROOT / "configs/train/libero/joint_lora_r32.yaml", [override])


def test_frozen_lora_warns_but_validates():
    exp = loader.load_train(ROOT / "configs/train/abc130k/folding_action_only.yaml")
    assert any("zero initialization" in w for w in exp.warnings)
    assert any("ignored" in w for w in exp.warnings)


def test_dual_resolution_requires_batch_sizes():
    with pytest.raises(ConfigError, match="video_batch_size"):
        loader.load_train(
            ROOT / "configs/train/libero/joint_lora_r32.yaml", ["data.video_num_frames=93"]
        )


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
