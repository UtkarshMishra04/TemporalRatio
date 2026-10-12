"""Translate experiment configs into trainer / evaluator invocations.

Training: each stage becomes one ``torchrun -m temporal_ratio.training.<trainer> ...`` command whose
arguments are exactly those the paper's launch scripts passed (verified by golden tests).
Multi-stage recipes initialize stage k+1's DiT from the latest checkpoint of stage k
(``<output_dir>/<action_dataset>/<stage_exp_name>/<step>/video_model_full.pt``).

Inference: produces the arguments and ``CFG_*`` environment variables consumed by the LIBERO
evaluators in ``scripts/libero/``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from temporal_ratio.config.schema import (
    ConfigError,
    InferenceExperiment,
    LiberoEvalExperiment,
    StageSpec,
    TrainExperiment,
)

#: Placeholder resolved at run time to the previous stage's latest checkpoint directory.
PREVIOUS_STAGE_LATEST = "{previous_stage_latest}"

TRAINER_MODULES = {
    "standard": "temporal_ratio.training.train",
    "dual_resolution": "temporal_ratio.training.train_dual_resolution",
}


def _fmt(x: float | int) -> str:
    """Format numbers like the launch scripts wrote them (``1e-4``, ``5.6e-5``, ``0.01``, ``0``).

    Uses the shortest representation that round-trips exactly, so the trainer parses the very
    same float the config holds.
    """
    if isinstance(x, int):
        return str(x)
    if x == 0:
        return "0"
    if abs(x) >= 1e-3:
        return repr(x)
    for digits in range(17):
        s = f"{x:.{digits}e}"
        if float(s) == x:
            break
    mantissa, exponent = s.split("e")
    return f"{mantissa}e{int(exponent)}"


@dataclass
class StageCommand:
    """A resolved trainer invocation for one stage."""

    stage: str
    module: str
    args: list[str]
    checkpoint_dir: Path
    #: ``(arg index, kind)`` of values that depend on the previous stage's latest checkpoint;
    #: kind ``video`` -> ``<latest>/video_model_full.pt``, kind ``action`` -> ``<latest>``.
    pending: list[tuple[int, str]] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    def torchrun(self, exp: TrainExperiment) -> list[str]:
        d = exp.distributed
        # ``python -m torch.distributed.run`` == ``torchrun``, pinned to the current interpreter.
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nnodes={d.nnodes}",
            f"--nproc_per_node={d.nproc_per_node}",
            f"--master_addr={d.master_addr}",
            f"--master_port={d.master_port}",
            f"--node_rank={d.node_rank}",
            "-m",
            self.module,
            *self.args,
        ]

    def resolve_previous_stage(self, previous_latest: Path) -> None:
        for index, kind in self.pending:
            self.args[index] = str(previous_latest / "video_model_full.pt") if kind == "video" else str(previous_latest)
        self.pending = []


def stage_checkpoint_dir(exp: TrainExperiment, stage: StageSpec) -> Path:
    return Path(exp.output_dir) / exp.data.action_dataset / exp.stage_exp_name(stage)


def latest_step_dir(ckpt_dir: Path) -> Path:
    steps = [int(p.name) for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit()] if ckpt_dir.exists() else []
    if not steps:
        raise FileNotFoundError(f"no step checkpoints found in {ckpt_dir}")
    return ckpt_dir / str(max(steps))


def build_stage_args(exp: TrainExperiment, stage: StageSpec) -> StageCommand:
    trainer = exp.resolve_trainer(stage)
    m, o, a, data = exp.model, exp.optim, exp.adaptation, exp.data
    args: list[str] = [
        data.action_dataset,
        "--exp_name", exp.stage_exp_name(stage),
        "--training_mode", stage.mode,
    ]
    if trainer == "dual_resolution":
        args += ["--video_num_frames", str(data.video_num_frames)]
        if data.video_frame_skip != 1:
            args += ["--video_frame_skip", str(data.video_frame_skip)]
        args += [
            "--video_batch_size", str(o.video_batch_size),
            "--action_batch_size", str(o.action_batch_size),
        ]
    args += ["--max_train_steps", str(stage.steps)]
    if trainer == "standard":
        args += ["--batch_size", str(o.batch_size)]
    args += [
        "--gradient_accumulation_steps", str(o.gradient_accumulation_steps),
        "--video_lr", _fmt(stage.video_lr),
        "--action_lr", _fmt(stage.action_lr),
        "--warmup_ratio", _fmt(o.warmup_ratio),
        "--min_lr_ratio", _fmt(o.min_lr_ratio),
        "--lr_schedule", o.lr_schedule,
        "--video_loss_weight", _fmt(stage.video_loss_weight),
        "--feature_layer", str(m.feature_layer),
        "--observation_horizon", str(m.observation_horizon),
        "--action_horizon", str(m.action_horizon),
        "--action_dim", str(m.action_dim),
        "--ema_decay", _fmt(o.ema_decay),
        "--ema_type", o.ema_type,
        "--weight_decay", _fmt(o.weight_decay),
        "--beta2", _fmt(o.beta2),
        "--gradient_clip_norm", _fmt(o.gradient_clip_norm),
        "--seed", str(exp.seed),
    ]
    if m.state_dim is not None:
        args += ["--state_dim", str(m.state_dim)]
    # LoRA adapters are injected whenever the config asks for LoRA, except in frozen_video stages
    # that load a previous stage's merged DiT (exactly as the paper launchers did). In a frozen_video
    # stage that starts from the base DiT the adapters are inert (see TrainExperiment.validate).
    lora_stage = not (stage.mode == "frozen_video" and stage.init_video_from != "base")
    if a.video == "lora" and lora_stage:
        args += [
            "--lora_enabled",
            "--lora_rank", str(a.lora.rank),
            "--lora_alpha", str(a.lora.alpha),
            "--lora_target_modules", ",".join(a.lora.target_modules),
        ]
        if not a.lora.fp32_params:
            args.append("--no_fp32_lora_params")

    pending: list[tuple[int, str]] = []
    if stage.init_video_from == "base":
        args += ["--base_cosmos_ckpt", m.base_video_checkpoint]
    elif stage.init_video_from == "previous_stage":
        args += ["--base_cosmos_ckpt", PREVIOUS_STAGE_LATEST]
        pending.append((len(args) - 1, "video"))
    else:
        args += ["--base_cosmos_ckpt", stage.init_video_from]
    is_first = stage is exp.stages[0]
    action_init = stage.init_action_from or (m.action_head_init if is_first else None)
    if action_init == "previous_stage":
        if is_first:
            raise ConfigError(f"stage '{stage.name}': the first stage cannot init from previous_stage")
        args += ["--action_head_weight_path", PREVIOUS_STAGE_LATEST]
        pending.append((len(args) - 1, "action"))
    elif action_init:
        args += ["--action_head_weight_path", action_init]
    if data.video_dataset:
        args += ["--video_dataset", data.video_dataset]
    args += [
        "--save_interval", str(stage.save_interval),
        "--log_interval", str(stage.log_interval),
        "--video_log_interval", str(stage.video_log_interval),
        "--video_gen_steps", str(stage.video_gen_steps),
        "--video_autoregressive_steps", str(stage.video_autoregressive_steps),
        "--checkpoint_dir", exp.output_dir,
        "--project_name", exp.logging.project,
    ]
    if trainer == "dual_resolution" and stage.video_aux_every_k is not None:
        args += ["--video_aux_every_k", str(stage.video_aux_every_k)]
    if stage.keep_every:
        args += ["--keep_every", str(stage.keep_every)]
    if stage.action_first_frame_drop_prob > 0:
        args += ["--action_first_frame_drop_prob", _fmt(stage.action_first_frame_drop_prob)]
    gradient_checkpointing = o.gradient_checkpointing if stage.gradient_checkpointing is None else stage.gradient_checkpointing
    if not gradient_checkpointing:
        args.append("--no_gradient_checkpointing")
    if not exp.logging.wandb:
        args.append("--no_wandb")
    if exp.resume and not exp.overwrite:
        args.append("--resume")
    if exp.overwrite:
        args.append("--overwrite")
    return StageCommand(
        stage=stage.name,
        module=TRAINER_MODULES[trainer],
        args=args,
        checkpoint_dir=stage_checkpoint_dir(exp, stage),
        pending=pending,
    )


def data_env(exp: TrainExperiment) -> dict[str, str]:
    """Environment of the trainer processes implied by ``exp.data``."""
    env = {}
    if exp.data.lerobot_home:
        env["HF_LEROBOT_HOME"] = exp.data.lerobot_home
    if exp.data.norm_stats:
        env["NORM_STATS_PATH"] = exp.data.norm_stats
    return env


def build_train_commands(exp: TrainExperiment) -> list[StageCommand]:
    commands = [build_stage_args(exp, s) for s in exp.stages]
    for cmd in commands:
        cmd.env.update(data_env(exp))
    return commands


# --------------------------------------------------------------------------------------------
# Inference / evaluation
# --------------------------------------------------------------------------------------------


def guidance_env(inf: InferenceExperiment) -> dict[str, str]:
    """``CFG_*`` environment for ``scripts/eval/eval_libero_guided.py``."""
    g = inf.guidance
    if g.type == "none":
        return {}
    if g.type == "flat":
        mode = "flat" if g.space == "velocity" else "action_flat"
    else:
        mode = g.tr_baseline
    env = {
        "CFG_W_BASE_LANG": repr(float(g.w_lang)),
        "CFG_W_BASE_PLAN": repr(float(g.w_plan)),
        "CFG_PLAN_LAT": str(g.plan_latent_frames),
        "CFG_MAX_SCALE": repr(float(g.s_max)),
        "CFG_TR_LAYER": str(g.tr_layer),
        "CFG_TR_LOG_LAYERS": ",".join(str(x) for x in inf.tr_log_layers),
        "CFG_TR_BASELINE_MODE": mode,
        "CFG_FORMULA": g.formula,
        "CFG_LEGACY_FLAT_FLOOR": "1" if g.legacy_flat_floor else "0",
    }
    if g.tr_baseline_fixed is not None:
        env["CFG_TR_BASELINE_FIXED"] = repr(float(g.tr_baseline_fixed))
    return env


@dataclass
class EvalCommand:
    script: str
    args: list[str]
    env: dict[str, str]


def build_libero_eval(exp: LiberoEvalExperiment, seed: int) -> EvalCommand:
    inf, ev = exp.inference, exp.eval
    script = "scripts/eval/eval_libero.py" if inf.guidance.type == "none" else "scripts/eval/eval_libero_guided.py"
    if not inf.checkpoint:
        raise ConfigError("inference.checkpoint is required")
    feature = f"_feature_{inf.feature_layer}" if inf.feature_layer is not None else ""
    args = [
        "--config", inf.policy_config,
        "--mode", f"{inf.policy_config}{feature}_diffsteps_{inf.video_denoising_steps}",
        "--inference_denoising_steps", str(inf.feature_extraction_step),
        "--replan_steps", str(inf.replan_every),
        "--num_trials_per_task", str(ev.num_trials_per_task),
        "--num_steps_wait", str(ev.num_steps_wait),
        "--seed", str(seed),
        "--video_out_path", str(Path(ev.output_dir) / f"seed{seed}"),
        "--checkpoint_step", Path(inf.checkpoint.rstrip("/")).name,  # only names the output directory
    ]
    if ev.split == "ood":
        args.append("--eval_ood")
    elif ev.split == "libero_90":
        args.append("--eval_90")
    elif ev.split == "all":
        args.append("--eval_all")
    if ev.suites:
        args += ["--task_suites", *ev.suites]
    if ev.task_ids:
        args += ["--task_ids", *map(str, ev.task_ids)]
    if ev.save_rollout_data:
        args.append("--save_rollout_data")
    if inf.video_weights == "full":
        args += ["--vid2world_ckpt", str(Path(inf.checkpoint) / "video_model_full.pt")]
    elif inf.video_weights == "lora":
        args += ["--vid2world_ckpt", inf.checkpoint]
    env = {"CHECKPOINT_DIR": inf.checkpoint, **guidance_env(inf)}
    if inf.norm_stats:
        env["NORM_STATS_PATH"] = inf.norm_stats
    if not inf.generate_video:
        env["TR_SKIP_VIDEO_DECODE"] = "1"
    return EvalCommand(script=script, args=args, env=env)
