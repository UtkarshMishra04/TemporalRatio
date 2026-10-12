"""Build a policy from an inference config (``configs/infer/*``).

Shared by ``trvam infer`` (offline evaluation), ``scripts/serve_policy.py`` (websocket server) and
the LIBERO evaluators, so all of them load checkpoints identically::

    exp = loader.load_inference("configs/infer/libero/no_guidance.yaml", ["checkpoint=..."])
    policy = load_policy(exp)                      # VideoModelPolicyWrapper (or guided wrapper)
    out = policy.infer(obs, **infer_kwargs(exp))   # actions, future_video, temporal_ratio, ...
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

from openpi_client import base_policy as _base_policy

from temporal_ratio.config.schema import ConfigError, InferenceExperiment

log = logging.getLogger(__name__)

#: Video-setup names recorded in older checkpoints' metadata.
LEGACY_SETUP_NAMES = {"xdof_flat_short": "abc130k_flat_short"}


def resolve_checkpoint(spec: str) -> pathlib.Path:
    """Local checkpoint step directory; ``hf://<owner>/<repo>/<subdir>`` is downloaded first."""
    if not spec.startswith("hf://"):
        path = pathlib.Path(spec).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"checkpoint directory {path} does not exist")
        return path
    from huggingface_hub import snapshot_download

    parts = spec[len("hf://"):].strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"expected hf://<owner>/<repo>[/<subdir>], got {spec}")
    repo_id, subdir = "/".join(parts[:2]), "/".join(parts[2:])
    root = snapshot_download(repo_id, allow_patterns=[f"{subdir}/*"] if subdir else None)
    return pathlib.Path(root) / subdir


def load_policy(exp: InferenceExperiment):
    """Load the checkpoint described by ``exp`` and wrap it for inference."""
    from temporal_ratio.config import registry
    from temporal_ratio.policies import factory, video_policy

    train_config = registry.get_config(exp.policy_config)
    checkpoint = resolve_checkpoint(exp.checkpoint)
    video_dit = (factory.read_metadata(checkpoint).get("config") or {}).get("video_dit") or {}

    setup = train_config.video_setup
    trained_setup = video_dit.get("setup")
    trained_setup = LEGACY_SETUP_NAMES.get(trained_setup, trained_setup)
    if trained_setup and trained_setup != setup:
        raise ConfigError(
            f"checkpoint was trained with video setup '{trained_setup}' but policy_config "
            f"'{exp.policy_config}' uses '{setup}'"
        )
    feature_layer = exp.feature_layer if exp.feature_layer is not None else int(video_dit.get("feature_layer", 19))

    policy = factory.create_trained_policy(
        train_config,
        checkpoint,
        norm_stats=exp.norm_stats,
        pytorch_device=exp.device,
        sample_kwargs={"num_steps": exp.action_denoising_steps},
    )
    wrapper = video_policy.VideoModelPolicyWrapper(
        policy,
        device=exp.device,
        inference_denoising_steps=exp.feature_extraction_step,
        vid2world_mode=setup,
        is_causal=bool(video_dit.get("causal_temporal", False)),
        feature_id=feature_layer,
        num_global_denoising_steps=exp.video_denoising_steps,
        vid2world_ckpt=str(video_checkpoint(checkpoint, exp.video_weights)),
        video_guidance=exp.video_guidance,
    )
    log.info(
        f"Loaded {checkpoint} | config={exp.policy_config} setup={setup} feature_layer={feature_layer} "
        f"video steps={exp.video_denoising_steps} (features after {exp.feature_extraction_step})"
    )
    if exp.guidance.type == "none":
        return wrapper
    return _guided(wrapper, exp)


def video_checkpoint(checkpoint: pathlib.Path, video_weights: str) -> pathlib.Path:
    """DiT weights of a checkpoint: merged ``video_model_full.pt`` or base + ``lora_adapters/``."""
    full = checkpoint / "video_model_full.pt"
    lora = checkpoint / "lora_adapters" / "adapter_config.json"
    if video_weights == "full" or (video_weights == "auto" and full.exists()):
        if not full.exists():
            raise FileNotFoundError(f"{full} not found")
        return full
    if video_weights in ("lora", "auto") and lora.exists():
        return checkpoint  # VideoProcessor applies the adapters to the pretrained base
    raise FileNotFoundError(f"{checkpoint} has neither video_model_full.pt nor lora_adapters/")


def _guided(wrapper, exp: InferenceExperiment):
    from temporal_ratio.guidance.tr_adaptive_policy import AdaptiveActionCFGWrapper

    g = exp.guidance
    if g.type == "flat":
        mode = "flat" if g.space == "velocity" else "action_flat"
    else:
        mode = g.tr_baseline
    return AdaptiveActionCFGWrapper(
        wrapper,
        w_base_lang=g.w_lang,
        w_base_plan=g.w_plan,
        plan_latent_frames=g.plan_latent_frames,
        max_scale=g.s_max,
        tr_layer=g.tr_layer,
        tr_baseline_mode=mode,
        tr_baseline_fixed=g.tr_baseline_fixed,
        cfg_formula=g.formula,
        tr_log_layers=list(exp.tr_log_layers),
        legacy_flat_floor=g.legacy_flat_floor,
        decode_video=exp.generate_video,
    )


def infer_kwargs(exp: InferenceExperiment, policy: Any) -> dict[str, Any]:
    """Keyword arguments for ``policy.infer`` implied by the config."""
    from temporal_ratio.policies.video_policy import VideoModelPolicyWrapper

    if isinstance(policy, VideoModelPolicyWrapper):
        return {"generate_video": exp.generate_video, "tr_layers": list(exp.tr_log_layers) or None}
    return {}


class ConfiguredPolicy(_base_policy.BasePolicy):
    """``infer(obs)`` with the config's options bound (for the websocket server)."""

    def __init__(self, exp: InferenceExperiment):
        self._exp = exp
        self._policy = load_policy(exp)
        self._kwargs = infer_kwargs(exp, self._policy)

    def infer(self, obs: dict) -> dict:  # type: ignore[override]
        out = self._policy.infer(obs, **self._kwargs)
        if out.get("future_video") is None:
            out.pop("future_video", None)
        out.pop("features", None)
        if "temporal_ratio" in out:  # msgpack clients require string map keys
            out["temporal_ratio"] = {str(k): v for k, v in out["temporal_ratio"].items()}
        return out

    def reset(self) -> None:
        if hasattr(self._policy, "reset_episode"):
            self._policy.reset_episode()

    @property
    def metadata(self) -> dict[str, Any]:
        return {"policy_config": self._exp.policy_config, "checkpoint": self._exp.checkpoint}
