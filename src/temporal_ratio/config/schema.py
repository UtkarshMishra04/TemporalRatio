"""Typed experiment configuration for training and inference.

One YAML file fully describes an experiment. It is validated against these dataclasses and
translated to the underlying trainer / evaluator invocations by :mod:`temporal_ratio.config.translate`.

Model family: Cosmos-Predict2.5-2B video DiT + Gemma-300M flow-matching action head that reads
DiT hidden states from ``model.feature_layer``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal

# --------------------------------------------------------------------------------------------
# Shared vocabulary
# --------------------------------------------------------------------------------------------

#: Trainer modes implemented by ``temporal_ratio.models.video_action_model.TrainingMode``.
TrainingMode = Literal["joint_full", "joint_detach", "frozen_video", "video_only"]

#: Video-DiT linear layers that LoRA can target (names inside each Cosmos DiT block).
LORA_TARGETABLE_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "output_proj",
    "mlp.layer1",
    "mlp.layer2",
)

#: Default LoRA targets used for every LoRA model in the paper.
DEFAULT_LORA_TARGETS: tuple[str, ...] = LORA_TARGETABLE_MODULES


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


# --------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------


@dataclass
class LoRASpec:
    """Low-rank adaptation of the video DiT."""

    rank: int = 32
    alpha: int = 32
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_LORA_TARGETS))
    #: Keep LoRA parameters in fp32 (trainer default).
    fp32_params: bool = True


@dataclass
class AdaptationSpec:
    """How the video backbone is adapted. The action head is always trained in full."""

    #: ``lora``: train LoRA adapters on the DiT; ``full``: full fine-tuning of the DiT;
    #: ``frozen``: never update the DiT (action head only).
    video: Literal["lora", "full", "frozen"] = "lora"
    lora: LoRASpec = field(default_factory=LoRASpec)


@dataclass
class ModelSpec:
    """Model architecture and initialization."""

    video_backbone: Literal["cosmos-predict2.5-2b"] = "cosmos-predict2.5-2b"
    action_head: Literal["gemma-300m"] = "gemma-300m"
    #: Where the first stage's DiT weights come from: ``base`` (Cosmos-Predict2.5-2B from the
    #: Hugging Face cache) or a path to a ``video_model_full.pt``.
    base_video_checkpoint: str = "base"
    #: Optional checkpoint step directory (containing ``model.safetensors``) to initialize the
    #: action head from, e.g. to fine-tune a released checkpoint.
    action_head_init: str | None = None
    #: DiT block whose hidden states feed the action head (paper models use 19).
    feature_layer: int = 19
    action_dim: int = 32
    action_horizon: int = 17
    observation_horizon: int = 17
    state_dim: int | None = None


@dataclass
class DataSpec:
    """Datasets.

    ``action_dataset`` is a named config of ``config/registry.py``; ``video_dataset`` a mixture of
    ``data/video_datasets.py``.
    """

    action_dataset: str = "videoVLA_libero_short"
    #: Dataset used for the video loss (``libero``, ``abc130k_pp24``, ...). ``None`` = same as action.
    video_dataset: str | None = None
    #: Frames per video clip for the video loss. 17 = same clip as the action branch;
    #: 93 = long-horizon auxiliary video clips (dual-resolution trainer).
    video_num_frames: int = 17
    video_frame_skip: int = 1
    #: Root of the LeRobot datasets (``$HF_LEROBOT_HOME``); ``None`` = the environment's.
    lerobot_home: str | None = None
    #: Normalization statistics: a ``norm_stats.json`` or a directory containing exactly one (e.g.
    #: ``<checkpoint>/assets`` to keep a fine-tuned model's statistics). ``None`` = the statistics in
    #: ``assets/<action_dataset>/`` (``scripts/data/compute_norm_stats.py``).
    norm_stats: str | None = None


@dataclass
class OptimSpec:
    """Optimizer and schedule (shared by all stages unless a stage overrides LRs)."""

    batch_size: int = 8
    #: Dual-resolution trainer only: per-GPU batch for long video clips / action clips.
    video_batch_size: int | None = None
    action_batch_size: int | None = None
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.01
    min_lr_ratio: float = 0.005
    lr_schedule: Literal["cosine", "cosmos"] = "cosine"
    weight_decay: float = 1e-8
    beta2: float = 0.99
    gradient_clip_norm: float = 1.0
    ema_decay: float = 0.999
    ema_type: Literal["power", "constant"] = "power"
    gradient_checkpointing: bool = True


@dataclass
class StageSpec:
    """One training stage = one trainer invocation."""

    name: str
    mode: TrainingMode
    steps: int
    video_lr: float
    action_lr: float
    video_loss_weight: float = 5.0
    #: ``auto`` picks the dual-resolution trainer when ``data.video_num_frames != 17``.
    trainer: Literal["auto", "standard", "dual_resolution"] = "auto"
    #: DiT initialization for this stage: ``base`` (model.base_video_checkpoint),
    #: ``previous_stage`` (the last checkpoint of the previous stage), or a path.
    init_video_from: str = "previous_stage"
    #: Action-head initialization for this stage: ``None`` (first stage: ``model.action_head_init``;
    #: later stages: fresh), ``previous_stage``, or a checkpoint step directory with model.safetensors.
    init_action_from: str | None = None
    #: Dual-resolution trainer: run the long-video loss every k steps.
    video_aux_every_k: int | None = None
    save_interval: int = 500
    log_interval: int = 10
    video_log_interval: int = 4000
    video_gen_steps: int = 6
    video_autoregressive_steps: int = 10
    #: Overrides ``optim.gradient_checkpointing`` for this stage (``None`` = inherit).
    gradient_checkpointing: bool | None = None
    #: Per action sample, probability of hiding the first latent frame (the clean conditioning
    #: frame) from the action head, so that it relies on the predicted frames; 0 = off.
    action_first_frame_drop_prob: float = 0.0
    #: Per action sample, probability of zeroing the robot state fed to the action head; 0 = off.
    state_dropout_rate: float = 0.0
    #: Also save at every multiple of this many steps (e.g. one epoch) and never delete those
    #: checkpoints; the others are pruned to the 5 most recent.
    keep_every: int | None = None


@dataclass
class LoggingSpec:
    wandb: bool = True
    project: str = "temporal-ratio"


@dataclass
class DistributedSpec:
    nnodes: int = 1
    nproc_per_node: int = 8
    master_addr: str = "localhost"
    master_port: int = 29500
    node_rank: int = 0


@dataclass
class TrainExperiment:
    """A complete training experiment."""

    name: str
    #: ``joint``: one joint_full stage; ``action_only``: one frozen_video stage (action head on
    #: frozen DiT features); ``two_stage``: video_only -> frozen_video; ``three_stage``:
    #: video_only -> joint_full -> frozen_video (93-frame recipe); ``custom``: any sequence.
    paradigm: Literal["joint", "action_only", "two_stage", "three_stage", "custom"]
    stages: list[StageSpec]
    model: ModelSpec = field(default_factory=ModelSpec)
    adaptation: AdaptationSpec = field(default_factory=AdaptationSpec)
    data: DataSpec = field(default_factory=DataSpec)
    optim: OptimSpec = field(default_factory=OptimSpec)
    logging: LoggingSpec = field(default_factory=LoggingSpec)
    distributed: DistributedSpec = field(default_factory=DistributedSpec)
    seed: int = 42
    output_dir: str = "checkpoints"
    #: Continue from the latest checkpoint of the run directory, if any (see docs/training.md).
    resume: bool = True
    #: Delete existing checkpoints of the run and start over; takes precedence over ``resume``.
    #: Without either, a run whose directory has checkpoints fails instead of overwriting them.
    overwrite: bool = False
    description: str = ""

    # ---------------------------------------------------------------------------------
    def validate(self) -> TrainExperiment:
        """Check cross-field consistency. Returns ``self`` for chaining.

        Hard inconsistencies raise :class:`ConfigError`; settings that are legal but have no
        effect are collected in :attr:`warnings`.
        """
        self.warnings: list[str] = []
        if not self.stages:
            raise ConfigError("at least one stage is required")
        names = [s.name for s in self.stages]
        if len(set(names)) != len(names):
            raise ConfigError(f"stage names must be unique, got {names}")

        expected: dict[str, list[str]] = {
            "joint": ["joint_full"],
            "action_only": ["frozen_video"],
            "two_stage": ["video_only", "frozen_video"],
            "three_stage": ["video_only", "joint_full", "frozen_video"],
        }
        modes = [s.mode for s in self.stages]
        if self.paradigm in expected and modes != expected[self.paradigm]:
            raise ConfigError(
                f"paradigm '{self.paradigm}' requires stage modes {expected[self.paradigm]}, got {modes}"
            )

        lora = self.adaptation.lora
        if self.adaptation.video == "lora":
            if lora.rank <= 0 or lora.alpha <= 0:
                raise ConfigError("LoRA rank and alpha must be positive")
            bad = sorted(set(lora.target_modules) - set(LORA_TARGETABLE_MODULES))
            if bad or not lora.target_modules:
                raise ConfigError(
                    f"invalid LoRA target_modules {bad or '[]'}; choose from {list(LORA_TARGETABLE_MODULES)}"
                )
        if self.adaptation.video == "frozen":
            for s in self.stages:
                if s.mode != "frozen_video":
                    raise ConfigError("adaptation.video=frozen requires every stage to use mode=frozen_video")

        for i, s in enumerate(self.stages):
            if i == 0 and s.init_video_from == "previous_stage":
                # The first stage starts from the model's base checkpoint.
                s.init_video_from = "base"
            if s.steps <= 0:
                raise ConfigError(f"stage '{s.name}': steps must be positive")
            if s.video_lr < 0 or s.action_lr < 0:
                raise ConfigError(f"stage '{s.name}': learning rates must be >= 0")
            if s.keep_every is not None and s.keep_every <= 0:
                raise ConfigError(f"stage '{s.name}': keep_every must be positive")
            if not 0 <= s.action_first_frame_drop_prob < 1:
                raise ConfigError(f"stage '{s.name}': action_first_frame_drop_prob must be in [0, 1)")
            if not 0 <= s.state_dropout_rate < 1:
                raise ConfigError(f"stage '{s.name}': state_dropout_rate must be in [0, 1)")
            if s.mode == "frozen_video":
                if s.video_lr != 0:
                    self.warnings.append(
                        f"stage '{s.name}': video_lr={s.video_lr} is ignored (frozen_video never updates the DiT)"
                    )
                if self.adaptation.video == "lora" and s.init_video_from == "base":
                    self.warnings.append(
                        f"stage '{s.name}': LoRA adapters are injected but stay at their zero initialization "
                        "(frozen_video); the action head trains on base-DiT features"
                    )
            if s.mode == "video_only" and s.video_loss_weight <= 0:
                raise ConfigError(f"stage '{s.name}': video_only requires video_loss_weight > 0")
            if self.resolve_trainer(s) == "dual_resolution":
                if self.optim.video_batch_size is None or self.optim.action_batch_size is None:
                    raise ConfigError(
                        f"stage '{s.name}' uses the dual-resolution trainer; set optim.video_batch_size "
                        "and optim.action_batch_size"
                    )
        if self.model.feature_layer < 0:
            raise ConfigError("model.feature_layer must be >= 0")
        return self

    def resolve_trainer(self, stage: StageSpec) -> Literal["standard", "dual_resolution"]:
        if stage.trainer != "auto":
            return stage.trainer
        return "dual_resolution" if self.data.video_num_frames != 17 else "standard"

    def stage_exp_name(self, stage: StageSpec) -> str:
        return self.name if len(self.stages) == 1 else f"{self.name}_{stage.name}"


# --------------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------------


@dataclass
class GuidanceSpec:
    """Action-level classifier-free guidance.

    The guided velocity is ``v = v_c + w_lang (v_c - v_u) + w_plan (v_plan - v_c)`` where
    ``v_c`` is conditioned on the task text, ``v_u`` on empty text, and ``v_plan`` on features
    from a longer video rollout of ``plan_latent_frames`` latent frames.

    ``tr_adaptive`` scales both weights by ``s = clip(TR / TR_0 - 1, 0, s_max)`` where TR_0 is the
    episode's first Temporal Ratio (``tr_baseline=per_episode``).
    """

    type: Literal["none", "flat", "tr_adaptive"] = "none"
    #: Where guidance is applied: ``velocity`` (every flow step, paper method) or ``action``
    #: (on the final action chunk).
    space: Literal["velocity", "action"] = "velocity"
    formula: Literal["extrapolation", "standard"] = "extrapolation"
    w_lang: float = 0.1
    w_plan: float = 0.1
    plan_latent_frames: int = 24
    tr_layer: int = 13
    tr_baseline: Literal["per_episode", "fixed", "uncond_ratio"] = "per_episode"
    tr_baseline_fixed: float | None = None
    s_max: float = 2.0
    #: Original research-code behaviour: when the adaptive scale is ~0 the base (flat) weights are
    #: used instead of no guidance. ``False`` follows the method as written in the paper.
    legacy_flat_floor: bool = False


@dataclass
class OfflineEvalSpec:
    """Open-loop evaluation on recorded LeRobot episodes (``trvam infer``)."""

    #: LeRobot dataset ids (empty: the policy config's dataset).
    datasets: list[str] = field(default_factory=list)
    episodes: list[int] = field(default_factory=lambda: [0])
    #: Policy queries per episode (one every ``replan_every`` frames); ``None`` = whole episode.
    max_steps: int | None = None
    #: Also report the errors over the first ``first_k`` actions of each chunk (``first_k_*``).
    first_k: int | None = None
    output_dir: str = "results/infer"


@dataclass
class InferenceExperiment:
    """Policy inference (serving, offline evaluation, LIBERO closed loop)."""

    #: Checkpoint step directory (contains model.safetensors, assets/, and lora_adapters/ or
    #: video_model_full.pt). ``hf://<repo>/<subdir>`` downloads it first.
    checkpoint: str
    #: Registered policy/data config that matches the checkpoint.
    policy_config: str
    #: Optional explicit norm_stats.json (default: the one stored inside the checkpoint).
    norm_stats: str | None = None
    #: DiT weights: ``auto`` prefers the merged ``video_model_full.pt`` and falls back to the
    #: pretrained base + ``lora_adapters/``; ``full`` / ``lora`` force one of them.
    video_weights: Literal["auto", "full", "lora"] = "auto"
    #: DiT block whose features feed the action head (default: the checkpoint's, else 19).
    feature_layer: int | None = None
    #: Total video denoising steps; features are taken after ``feature_extraction_step`` of them.
    video_denoising_steps: int = 6
    feature_extraction_step: int = 2
    #: Flow-matching (Euler) steps of the action head.
    action_denoising_steps: int = 10
    #: Also run the remaining video denoising steps (with video CFG ``video_guidance``) and decode
    #: the predicted future frames. Off: video denoising stops at ``feature_extraction_step``.
    generate_video: bool = False
    video_guidance: float = 7.0
    replan_every: int = 10
    guidance: GuidanceSpec = field(default_factory=GuidanceSpec)
    #: Action-expert layers whose Temporal Ratio is logged (empty: no TR logging).
    tr_log_layers: list[int] = field(default_factory=lambda: [2, 5, 10, 13, 15])
    #: Root of the LeRobot datasets used by ``offline`` (``$HF_LEROBOT_HOME``); ``None`` = environment's.
    lerobot_home: str | None = None
    offline: OfflineEvalSpec = field(default_factory=OfflineEvalSpec)
    device: str = "cuda:0"
    seed: int = 7

    def validate(self) -> InferenceExperiment:
        if not self.checkpoint:
            raise ConfigError("checkpoint is required")
        if not 0 <= self.feature_extraction_step < self.video_denoising_steps:
            raise ConfigError("feature_extraction_step must be within [0, video_denoising_steps)")
        if self.action_denoising_steps <= 0:
            raise ConfigError("action_denoising_steps must be positive")
        if self.offline.max_steps is not None and self.offline.max_steps <= 0:
            raise ConfigError("offline.max_steps must be positive")
        if self.offline.first_k is not None and self.offline.first_k <= 0:
            raise ConfigError("offline.first_k must be positive")
        if self.replan_every <= 0:
            raise ConfigError("replan_every must be positive")
        g = self.guidance
        if g.type != "none" and g.w_lang == 0 and g.w_plan == 0:
            raise ConfigError("guidance enabled but w_lang == w_plan == 0")
        if g.type == "tr_adaptive" and g.space != "velocity":
            raise ConfigError("tr_adaptive guidance is applied in velocity space")
        if g.tr_baseline == "fixed" and g.tr_baseline_fixed is None:
            raise ConfigError("tr_baseline=fixed requires tr_baseline_fixed")
        if g.s_max <= 0:
            raise ConfigError("guidance.s_max must be positive")
        return self


@dataclass
class LiberoEvalSpec:
    """Closed-loop LIBERO evaluation."""

    split: Literal["id", "ood", "libero_90", "all"] = "id"
    #: Explicit suites (overrides ``split``), e.g. ``[libero_spatial]``; ``task_ids`` limits the tasks.
    suites: list[str] = field(default_factory=list)
    task_ids: list[int] = field(default_factory=list)
    num_trials_per_task: int = 10
    seeds: list[int] = field(default_factory=lambda: [7])
    num_steps_wait: int = 10
    output_dir: str = "results/libero"
    save_rollout_data: bool = False


@dataclass
class LiberoEvalExperiment:
    inference: InferenceExperiment
    eval: LiberoEvalSpec = field(default_factory=LiberoEvalSpec)

    def validate(self) -> LiberoEvalExperiment:
        self.inference.validate()
        if self.eval.num_trials_per_task <= 0:
            raise ConfigError("eval.num_trials_per_task must be positive")
        return self


def as_dict(obj: object) -> dict:
    """Plain-dict view of a config dataclass (for printing / provenance)."""
    return dataclasses.asdict(obj)  # type: ignore[arg-type]
