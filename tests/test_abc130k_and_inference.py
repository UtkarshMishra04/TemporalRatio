from __future__ import annotations

from pathlib import Path

import pytest

from temporal_ratio.config import loader
from temporal_ratio.config.schema import ConfigError
from temporal_ratio.data import abc130k

ROOT = Path(__file__).resolve().parents[1]


def test_abc130k_task_lists():
    assert len(abc130k.TASKS) == 22 == len(set(abc130k.TASKS))
    assert set(abc130k.TASKS) == set(abc130k.PP24_NAMES)
    assert sum(len(t) for t in abc130k.SKILLS.values()) == 22
    assert abc130k.repo_ids("folding") == ("abc130k/fold_and_stack_the_t_shirts",)
    assert len(set(abc130k.repo_ids())) == 22
    assert abc130k.repo_ids("fold_pen_bottle") == (
        "abc130k/fold_and_stack_the_t_shirts", "abc130k/open_the_pen_caps", "abc130k/unscrew_the_bottle_caps",
    )
    assert abc130k.repo_id("open_the_pen_caps", split="val") == "abc130k/open_the_pen_caps_val"
    # single-task groups of the fold / pen / bottle tasks
    assert abc130k.repo_ids("folding") + abc130k.repo_ids("pen_caps") + abc130k.repo_ids("assembly") == \
        abc130k.repo_ids("fold_pen_bottle")
    assert not set(abc130k.EXTRA_TASKS) & set(abc130k.PP24_NAMES)
    with pytest.raises(KeyError):
        abc130k.repo_id("handoff")


def test_inference_defaults_follow_checkpoint():
    exp = loader.load_inference(ROOT / "configs/infer/abc130k/yam_pp24.yaml")
    assert exp.feature_layer is None  # taken from the checkpoint metadata
    assert exp.video_weights == "auto"
    assert exp.action_denoising_steps == 10
    with pytest.raises(ConfigError, match="feature_extraction_step"):
        loader.load_inference(ROOT / "configs/infer/abc130k/yam_pp24.yaml", ["feature_extraction_step=6"])


def test_norm_stats_lookup(tmp_path):
    from temporal_ratio.data import normalize

    (tmp_path / "assets" / "org" / "task").mkdir(parents=True)
    (tmp_path / "assets" / "org" / "task" / "norm_stats.json").write_text("{}")
    assert normalize.find(tmp_path / "assets") == tmp_path / "assets" / "org" / "task"
    assert normalize.find(tmp_path / "assets" / "org" / "task" / "norm_stats.json") == tmp_path / "assets" / "org" / "task"
    (tmp_path / "assets" / "other").mkdir()
    (tmp_path / "assets" / "other" / "norm_stats.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="exactly one"):
        normalize.find(tmp_path / "assets")
