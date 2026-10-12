"""Golden tests: every training preset reproduces the trainer invocation of the original run.

``launcher_argv.json`` holds the trainer arguments of the original launch scripts (extracted from
the shell scripts with their variables substituted; the trainers were then called
``openpi.training.train_joint`` / ``openpi.training.train_joint_93f``). Both argument lists are
parsed with the trainer's own argparse parser and the resulting namespaces must be identical,
except for:

  * names that the release changes (experiment name, checkpoint dir, W&B project, and the
    dataset config names of the ABC-130k LeRobot conversion), and
  * ``--text_dropout_rate``, which the trainer accepts but ignores (text dropout is applied by the
    Cosmos conditioner at 0.2 in every run).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from temporal_ratio.config import loader, translate
from temporal_ratio.training import train_args, train_dual_resolution_args

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "train"
LAUNCHERS = json.loads((Path(__file__).parent / "launcher_argv.json").read_text())
IGNORED = {"exp_name", "checkpoint_dir", "project_name", "text_dropout_rate"}
MODULES = {
    "openpi.training.train_joint": "temporal_ratio.training.train",
    "openpi.training.train_joint_93f": "temporal_ratio.training.train_dual_resolution",
}
PARSERS = {
    "temporal_ratio.training.train": train_args.build_arg_parser,
    "temporal_ratio.training.train_dual_resolution": train_dual_resolution_args.build_arg_parser,
}
ABC_PP24 = {
    "data_config_name": ("videoVLA_xdof_multi_task_short", "videoVLA_abc130k_pp24_short"),
    "video_dataset": ("xdof_10", "abc130k_pp24"),
}
ABC_FOLDING = {
    "data_config_name": ("videoVLA_xdof_folding_short", "videoVLA_abc130k_folding_short"),
    "video_dataset": ("xdof_folding", "abc130k_folding"),
}

# (preset, stage index, launcher command, renamed fields {dest: (launcher value, release value)})
GOLDEN = [
    ("libero/joint_lora_r16.yaml", 0, "v3_libero/joint_r16.sh#0", {}),
    ("libero/joint_lora_r32.yaml", 0, "v3_libero/joint_r32.sh#0", {}),
    ("libero/joint_lora_r64.yaml", 0, "v3_libero/joint_r64.sh#0", {}),
    ("libero/joint_full.yaml", 0, "v3_libero/full_joint.sh#0", {}),
    ("libero/joint_93f_lora_r16.yaml", 0, "v3_libero/joint_93f_r16.sh#0", {}),
    ("libero/joint_93f_lora_r32.yaml", 0, "v3_libero/joint_93f_r32.sh#0", {}),
    ("libero/joint_93f_lora_r64.yaml", 0, "v3_libero/joint_93f_r64.sh#0", {}),
    ("libero/joint_93f_full.yaml", 0, "v3_libero/full_joint_93f.sh#0", {}),
    *[(f"libero/two_stage_lora_r{r}.yaml", i, f"v2_libero/2stage_r{r}.sh#{i}", {}) for r in (16, 32, 64) for i in (0, 1)],
    *[("libero/two_stage_full.yaml", i, f"v2_libero/full_2stage.sh#{i}", {}) for i in (0, 1)],
    *[("libero/two_stage_93f_lora_r32.yaml", i, f"v2_libero/2stage_93f_r32.sh#{i}", {}) for i in (0, 1)],
    *[("libero/two_stage_93f_full.yaml", i, f"v2_libero/full_2stage_93f.sh#{i}", {}) for i in (0, 1)],
    *[("libero/two_stage_lora_r32_16k.yaml", i, f"libero/joint/2b/train_2stage_lora_r32.sh#{i}", {}) for i in (0, 1)],
    *[("libero/three_stage_93f_lora_r32.yaml", i, f"joint_93f/run_lora_dropout.sh#{i}", {}) for i in (0, 1, 2)],
    ("abc130k/pp24_joint_lora_r32.yaml", 0, "v3_xdof10/joint_93f_r32.sh#0", ABC_PP24),
    ("abc130k/folding_action_only.yaml", 0, "v3_xdof10/skill_folding.sh#0", ABC_FOLDING),
    ("abc130k/skill_finetune_93f.yaml", 0, "v3_xdof10/skill_finetune.sh#0",
     {**ABC_FOLDING, "video_dataset": ("xdof_10", "abc130k_pp24")}),
]


def _ids() -> list[str]:
    return [f"{preset}[{stage}]" for preset, stage, *_ in GOLDEN]


@pytest.mark.parametrize(("preset", "stage", "launcher", "renames"), GOLDEN, ids=_ids())
def test_preset_matches_original_launcher(preset, stage, launcher, renames):
    exp = loader.load_train(CONFIGS / preset)
    cmd = translate.build_train_commands(exp)[stage]
    module = MODULES[LAUNCHERS[launcher]["module"]]
    launcher_argv = LAUNCHERS[launcher]["argv"]
    assert cmd.module == module

    parse = PARSERS[module]()
    ours = vars(parse.parse_args(cmd.args))
    theirs = vars(parse.parse_args(launcher_argv))
    for dest, (launcher_value, release_value) in renames.items():
        assert theirs[dest] == launcher_value
        assert ours[dest] == release_value
        ours[dest] = theirs[dest]

    diff = {
        k: (theirs.get(k), ours.get(k))
        for k in sorted(set(ours) | set(theirs))
        if k not in IGNORED and ours.get(k) != theirs.get(k)
    }
    assert diff == {}, f"{preset} stage {stage}: (launcher, release) differences: {diff}"


def test_every_train_preset_is_covered_or_derived():
    """Presets without a golden launcher must inherit (directly) from a golden-covered one."""
    assert {k for _, _, k, _ in GOLDEN} == set(LAUNCHERS), "every extracted launcher command is tested"
    covered = {p for p, *_ in GOLDEN}
    for path in sorted(CONFIGS.rglob("*.yaml")):
        rel = str(path.relative_to(CONFIGS))
        if path.name.startswith("_") or rel in covered:
            continue
        base = loader.yaml.safe_load(path.read_text()).get("_base_")
        parent = str((path.parent / base).resolve().relative_to(CONFIGS)) if isinstance(base, str) else None
        assert parent in covered or path.name == "finetune_from_checkpoint.yaml", (
            f"{rel} is not covered by a golden launcher test (base: {parent})"
        )


def test_multistage_placeholders_resolve(tmp_path):
    exp = loader.load_train(CONFIGS / "libero/three_stage_93f_lora_r32.yaml")
    cmds = translate.build_train_commands(exp)
    assert [c.pending for c in cmds][0] == []
    s2_latest = tmp_path / "s2" / "1234"
    s2_latest.mkdir(parents=True)
    cmds[2].resolve_previous_stage(s2_latest)
    args = cmds[2].args
    assert args[args.index("--base_cosmos_ckpt") + 1] == str(s2_latest / "video_model_full.pt")
    assert args[args.index("--action_head_weight_path") + 1] == str(s2_latest)
    assert translate.PREVIOUS_STAGE_LATEST not in args


def test_latest_step_dir(tmp_path):
    for step in ("500", "12000", "tmp_9", "3000"):
        (tmp_path / step).mkdir()
    assert translate.latest_step_dir(tmp_path).name == "12000"
    with pytest.raises(FileNotFoundError):
        translate.latest_step_dir(tmp_path / "missing")


@pytest.mark.parametrize("value", [1e-4, 5.6e-5, 2e-5, 1e-8, 0.005, 0.999, 3.0e-4, 1.23456789e-7, 0.0, 5.0])
def test_number_formatting_round_trips(value):
    assert float(translate._fmt(value)) == value
    assert translate._fmt(5.6e-5) == "5.6e-5" and translate._fmt(1e-4) == "1e-4"
