"""Typed inference configuration.

One YAML file fully describes how a checkpoint is loaded and queried (``trvam serve``,
:func:`temporal_ratio.inference.load_policy`). It is validated against these dataclasses.

Model family: Cosmos-Predict2.5-2B video DiT + Gemma-300M flow-matching action head that reads
DiT hidden states from ``feature_layer``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


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
class InferenceExperiment:
    """Policy inference (websocket server and Python API)."""

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
    guidance: GuidanceSpec = field(default_factory=GuidanceSpec)
    #: Action-expert layers whose Temporal Ratio is returned with each query (empty: none).
    tr_log_layers: list[int] = field(default_factory=lambda: [2, 5, 10, 13, 15])
    device: str = "cuda:0"

    def validate(self) -> InferenceExperiment:
        if not self.checkpoint:
            raise ConfigError("checkpoint is required")
        if not 0 <= self.feature_extraction_step < self.video_denoising_steps:
            raise ConfigError("feature_extraction_step must be within [0, video_denoising_steps)")
        if self.action_denoising_steps <= 0:
            raise ConfigError("action_denoising_steps must be positive")
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


def as_dict(obj: object) -> dict:
    """Plain-dict view of a config dataclass (for printing / provenance)."""
    return dataclasses.asdict(obj)  # type: ignore[arg-type]
