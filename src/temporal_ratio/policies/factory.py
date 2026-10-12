"""Create an inference policy (action head + transforms) from a training checkpoint."""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Any

import torch

from temporal_ratio.config import registry as _config
import temporal_ratio.data.normalize as _normalize
import temporal_ratio.data.transforms as transforms
import temporal_ratio.policies.video_policy as _policy

log = logging.getLogger(__name__)


def read_metadata(checkpoint_dir: pathlib.Path | str) -> dict[str, Any]:
    """``metadata.pt`` of a checkpoint step directory (empty dict if absent)."""
    path = pathlib.Path(checkpoint_dir) / "metadata.pt"
    if not path.exists():
        return {}
    return torch.load(path, map_location="cpu", weights_only=False)


def load_checkpoint_norm_stats(
    checkpoint_dir: pathlib.Path, asset_id: str | None, override: str | None = None
) -> dict[str, transforms.NormStats]:
    """Normalization statistics for a checkpoint.

    Lookup order: ``override`` (file or directory), ``$NORM_STATS_PATH``,
    ``<checkpoint>/assets/<asset_id>/norm_stats.json``, then the unique ``norm_stats.json``
    anywhere under ``<checkpoint>/assets`` (checkpoints trained under a different dataset id).
    """
    override = override or os.environ.get("NORM_STATS_PATH")
    if override:
        return _normalize.load(_normalize.find(override))
    assets = checkpoint_dir / "assets"
    if asset_id and (assets / asset_id / "norm_stats.json").exists():
        return _normalize.load(assets / asset_id)
    found = sorted(assets.rglob("norm_stats.json")) if assets.exists() else []
    if len(found) == 1:
        log.info(f"Using the checkpoint's norm stats {found[0].relative_to(checkpoint_dir)}")
        return _normalize.load(found[0].parent)
    raise FileNotFoundError(
        f"cannot determine norm stats for {checkpoint_dir}: expected assets/{asset_id}/norm_stats.json "
        f"or exactly one norm_stats.json under assets/ (found {len(found)}); set norm_stats explicitly"
    )


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | str | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create the action-head policy of a checkpoint.

    Args:
        train_config: named robot/dataset config (``config.registry``) matching the checkpoint;
            it defines the model dimensions and the input/output transforms.
        checkpoint_dir: checkpoint step directory with ``model.safetensors``.
        repack_transforms: transforms applied before all others.
        sample_kwargs: extra kwargs for ``sample_actions`` (e.g. ``num_steps``).
        default_prompt: prompt injected when the observation has none.
        norm_stats: statistics or a path to ``norm_stats.json``; default: from the checkpoint.
        pytorch_device: e.g. ``"cuda:0"``; default ``cuda`` if available.
    """
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    weight_path = checkpoint_dir / "model.safetensors"
    if not weight_path.exists():
        raise FileNotFoundError(f"{weight_path} not found (expected a PyTorch checkpoint step directory)")
    repack_transforms = repack_transforms or transforms.Group()

    metadata = read_metadata(checkpoint_dir)
    trained_with = (metadata.get("config") or {}).get("name")
    if trained_with and trained_with != train_config.name:
        log.info(f"Checkpoint was trained with config '{trained_with}'; using '{train_config.name}' for inference")
    try:
        action_expert_variant = metadata["config"]["model"]["action_expert_variant"]
    except (KeyError, TypeError):
        action_expert_variant = "gemma_300m"
    train_config = train_config.set_action_expert_config(action_expert_variant)

    log.info(f"Loading action head from {weight_path}")
    model = train_config.model.load_pytorch(train_config, str(weight_path))
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None or isinstance(norm_stats, str):
        norm_stats = load_checkpoint_norm_stats(checkpoint_dir, data_config.asset_id, norm_stats)

    if pytorch_device is None:
        pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        pytorch_device=pytorch_device,
    )
