from __future__ import annotations

from pathlib import Path

import pytest

from temporal_ratio.config import loader, translate
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


def test_eval_command_carries_feature_layer_and_video_weights():
    path = ROOT / "configs/eval/libero/id.yaml"
    exp = loader.load_libero_eval(path, ["inference.checkpoint=/c/7", "inference.feature_layer=17",
                                         "inference.video_weights=full"])
    cmd = translate.build_libero_eval(exp, seed=0)
    assert cmd.args[cmd.args.index("--mode") + 1] == "videoVLA_libero_short_feature_17_diffsteps_6"
    assert cmd.args[cmd.args.index("--vid2world_ckpt") + 1] == "/c/7/video_model_full.pt"


def test_converter_task_map_matches_package():
    import importlib.util
    import sys

    path = ROOT / "scripts/data/convert_abc130k_to_lerobot.py"
    spec = importlib.util.spec_from_file_location("convert_abc130k_to_lerobot", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    assert module.ABC_TO_PP24 == abc130k.PP24_NAMES
    parser = module.build_arg_parser()
    args = parser.parse_args(["--src", "hf:fold_and_stack_the_t_shirts"])
    assert args.hf_features_style == "sequence"  # readable by the training stack (datasets<3)


def test_eval_command_suite_selection():
    exp = loader.load_libero_eval(ROOT / "configs/eval/libero/id.yaml",
                                  ["inference.checkpoint=/c/1", "eval.suites=[libero_spatial]", "eval.task_ids=[0, 3]"])
    args = translate.build_libero_eval(exp, seed=0).args
    assert args[args.index("--task_suites") + 1] == "libero_spatial"
    assert args[args.index("--task_ids") + 1: args.index("--task_ids") + 3] == ["0", "3"]


def test_action_first_frame_drop_prob(monkeypatch):
    monkeypatch.setenv("TR_FINETUNE_FROM", "/ckpts/yam/480000")
    path = ROOT / "configs/train/abc130k/finetune_from_checkpoint.yaml"
    exp = loader.load_train(path, ["stages.0.action_first_frame_drop_prob=0.2"])
    args = translate.build_stage_args(exp, exp.stages[0]).args
    assert args[args.index("--action_first_frame_drop_prob") + 1] == "0.2"
    exp = loader.load_train(path)  # off by default
    assert "--action_first_frame_drop_prob" not in translate.build_stage_args(exp, exp.stages[0]).args
    with pytest.raises(ConfigError):
        loader.load_train(path, ["stages.0.action_first_frame_drop_prob=1"])


def test_errors_in_heldout_std_units():
    import numpy as np

    from temporal_ratio import offline

    pred, gt = np.zeros((50, 2)), np.zeros((50, 2))
    pred[:, 0] = 0.1  # 0.05 std off in dimension 0 (std 2), exact in dimension 1
    e = offline.std_errors(pred, gt, np.array([2.0, 1.0]))
    assert e == pytest.approx({"action_mse_std": 0.00125, "action_mae_std": 0.025, "first_action_mae_std": 0.025})
    summary = offline._std_summary([e, e])
    assert summary["action_rmse_std"] == pytest.approx(0.00125**0.5)
    pred[25:, 1] = 1.0  # error only after the first 25 actions
    e = offline.std_errors(pred, gt, np.array([2.0, 1.0]), first_k=25)
    assert e["first_k_mae_std"] == pytest.approx(0.025) and e["action_mae_std"] > 0.025
    summary = offline._std_summary([e], first_k=25)
    assert summary["first_k"] == 25 and summary["first_k_rmse_std"] == pytest.approx(0.00125**0.5)
    exp = loader.load_inference(ROOT / "configs/infer/abc130k/fold_pen_bottle_heldout.yaml", ["checkpoint=/c/1"])
    assert exp.replan_every == 150 and exp.offline.max_steps is None and len(exp.offline.episodes) == 20


def test_keep_every_resume_and_overwrite(monkeypatch):
    monkeypatch.setenv("TR_FINETUNE_FROM", "/ckpts/yam/480000")
    path = ROOT / "configs/train/abc130k/finetune_from_checkpoint.yaml"
    exp = loader.load_train(path, ["stages.0.keep_every=3633"])
    args = translate.build_stage_args(exp, exp.stages[0]).args
    assert args[args.index("--keep_every") + 1] == "3633"
    assert "--resume" in args and "--overwrite" not in args  # runs resume by default
    exp = loader.load_train(path, ["overwrite=true"])
    args = translate.build_stage_args(exp, exp.stages[0]).args
    assert "--overwrite" in args and "--resume" not in args
    with pytest.raises(ConfigError):
        loader.load_train(path, ["stages.0.keep_every=0"])


def test_train_cli_gpus_and_data_env(capsys, monkeypatch):
    from temporal_ratio import cli

    monkeypatch.setenv("TR_FINETUNE_FROM", "/ckpts/yam/878500")
    rc = cli.main(["train", str(ROOT / "configs/train/abc130k/finetune_from_checkpoint.yaml"), "--gpus", "1",
                   "--set", "data.lerobot_home=/data/abc130k", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "--nnodes=1" in out and "--nproc_per_node=1" in out
    assert "HF_LEROBOT_HOME=/data/abc130k" in out
    assert "NORM_STATS_PATH=/ckpts/yam/878500/assets" in out
    assert "videoVLA_abc130k_fold_pen_bottle_short" in out and "--video_dataset abc130k_fold_pen_bottle" in out


def test_offline_eval_section():
    exp = loader.load_inference(ROOT / "configs/infer/abc130k/fold_pen_bottle.yaml", ["checkpoint=/c/1"])
    assert exp.offline.datasets == [
        "abc130k/fold_and_stack_the_t_shirts_val", "abc130k/open_the_pen_caps_val", "abc130k/unscrew_the_bottle_caps_val",
    ]
    assert exp.offline.episodes == [0, 1, 2, 3, 4] and exp.offline.max_steps == 20
    with pytest.raises(ConfigError, match="offline.max_steps"):
        loader.load_inference(ROOT / "configs/infer/abc130k/fold_pen_bottle.yaml", ["offline.max_steps=0"])


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
