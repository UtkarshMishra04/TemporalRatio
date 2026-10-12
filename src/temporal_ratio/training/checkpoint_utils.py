"""Utilities for loading joint training checkpoints at inference time.

Handles the key mismatch between training (gradient checkpointing adds
``_checkpoint_wrapped_module`` to block keys) and inference (no wrapper).

Usage — video only:
    from temporal_ratio.training.checkpoint_utils import load_video_checkpoint
    video_processor = VideoProcessor(setup="libero_flat_short_causal", device="cuda")
    load_video_checkpoint(video_processor, "checkpoints/.../14000")

Usage — video + action:
    from temporal_ratio.training.checkpoint_utils import load_joint_checkpoint
    video_processor, action_model = ..., ...
    load_joint_checkpoint(video_processor, action_model, "checkpoints/.../14000")

Usage — resolve latest step automatically:
    from temporal_ratio.training.checkpoint_utils import load_joint_checkpoint, resolve_latest_step
    ckpt_dir = resolve_latest_step("checkpoints/.../joint_libero_repa_dino_clip")
    load_joint_checkpoint(video_processor, action_model, ckpt_dir)
"""

import logging
from pathlib import Path

import torch

log = logging.getLogger(__name__)


def _strip_checkpoint_wrapper(state_dict: dict) -> dict:
    """Strip ``_checkpoint_wrapped_module`` from keys.

    Gradient checkpointing during training wraps each block, adding
    ``._checkpoint_wrapped_module`` to parameter names. At inference
    the blocks are unwrapped, so these need to be removed.
    """
    cleaned = {}
    n_stripped = 0
    for k, v in state_dict.items():
        new_k = k.replace("._checkpoint_wrapped_module", "")
        if new_k != k:
            n_stripped += 1
        cleaned[new_k] = v
    if n_stripped:
        log.info(f"Stripped _checkpoint_wrapped_module from {n_stripped}/{len(state_dict)} keys")
    return cleaned


def resolve_latest_step(run_dir: str | Path) -> Path:
    """Find the latest checkpoint step directory inside a training run dir.

    Args:
        run_dir: Path like ``checkpoints/.../joint_libero_repa_dino_clip/``

    Returns:
        Path to the latest step directory, e.g. ``run_dir/14000``
    """
    run_dir = Path(run_dir)
    steps = [
        int(d.name) for d in run_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    if not steps:
        raise FileNotFoundError(f"No checkpoint steps found in {run_dir}")
    latest = max(steps)
    log.info(f"Resolved latest checkpoint: {run_dir / str(latest)} (from {len(steps)} steps)")
    return run_dir / str(latest)


def load_video_checkpoint(
    video_processor: torch.nn.Module,
    ckpt_dir: str | Path,
    device: str | torch.device = "cpu",
) -> int:
    """Load trained video model into a VideoProcessor.

    Supports two formats (prefers video_model_full.pt):
      - ``video_model_full.pt``: Full model, same format as base Cosmos checkpoint
        (keys: ``net.*``). Loaded into ``video_processor.model`` via the standard
        model loader path.
      - ``video_dit_blocks.pt``: Legacy partial format with only trainable block
        params (keys: ``model.net.blocks.*``). Needs checkpoint-wrapper stripping.

    Args:
        video_processor: A ``VideoProcessor`` instance.
        ckpt_dir: Path to a specific step directory (e.g. ``checkpoints/.../14000``).
        device: Device to load weights onto.

    Returns:
        Number of parameters loaded.
    """
    ckpt_dir = Path(ckpt_dir)

    # Prefer full model checkpoint
    full_path = ckpt_dir / "video_model_full.pt"
    legacy_path = ckpt_dir / "video_dit_blocks.pt"

    if full_path.exists():
        state = torch.load(full_path, map_location=device, weights_only=True)
        # video_model_full.pt has net.* keys — load into video_processor.model
        model = video_processor.model if hasattr(video_processor, 'model') else video_processor
        if hasattr(model, 'net'):
            # Strip "net." prefix to load directly into model.net
            net_state = {k[len("net."):]: v for k, v in state.items() if k.startswith("net.")}
            missing, unexpected = model.net.load_state_dict(net_state, strict=False)
            loaded = len(net_state) - len(unexpected)
            if unexpected:
                log.warning(f"Unexpected keys: {unexpected[:5]}...")
            log.info(f"Loaded {loaded}/{len(net_state)} video DiT params from {full_path}")
            return loaded
        else:
            # Fallback: try loading with net.* prefix directly
            missing, unexpected = model.load_state_dict(state, strict=False)
            loaded = len(state) - len(unexpected)
            log.info(f"Loaded {loaded}/{len(state)} video params from {full_path}")
            return loaded

    elif legacy_path.exists():
        state = torch.load(legacy_path, map_location=device, weights_only=True)
        state = _strip_checkpoint_wrapper(state)
        missing, unexpected = video_processor.load_state_dict(state, strict=False)
        loaded = len(state) - len(unexpected)
        if unexpected:
            log.warning(f"Unexpected keys: {unexpected[:5]}...")
        if loaded == 0:
            log.error(f"Loaded 0 video DiT params! Check key format in {legacy_path}")
        else:
            log.info(f"Loaded {loaded}/{len(state)} video DiT params from {legacy_path} (legacy)")
        return loaded

    else:
        raise FileNotFoundError(f"No video_model_full.pt or video_dit_blocks.pt in {ckpt_dir}")


def load_action_checkpoint(
    action_model: torch.nn.Module,
    ckpt_dir: str | Path,
    device: str | torch.device = "cpu",
    use_ema: bool = True,
) -> int:
    """Load trained action head weights into an action model.

    Args:
        action_model: A ``VideoVLAPytorch`` instance.
        ckpt_dir: Path to a specific step directory.
        device: Device to load weights onto.
        use_ema: If True, prefer ``action_model_ema.safetensors`` over
            ``action_model.safetensors``.

    Returns:
        Number of parameters loaded.
    """
    import safetensors.torch

    ckpt_dir = Path(ckpt_dir)

    # Prefer EMA weights for inference
    if use_ema and (ckpt_dir / "action_model_ema.safetensors").exists():
        path = ckpt_dir / "action_model_ema.safetensors"
    elif (ckpt_dir / "action_model.safetensors").exists():
        path = ckpt_dir / "action_model.safetensors"
    elif (ckpt_dir / "model.safetensors").exists():
        path = ckpt_dir / "model.safetensors"
    else:
        raise FileNotFoundError(f"No action model safetensors in {ckpt_dir}")

    safetensors.torch.load_model(action_model, path, device=str(device))
    n_params = sum(p.numel() for p in action_model.parameters())
    log.info(f"Loaded action model from {path} ({n_params/1e6:.0f}M params, ema={use_ema})")
    return n_params


def load_joint_checkpoint(
    video_processor: torch.nn.Module,
    action_model: torch.nn.Module | None,
    ckpt_dir: str | Path,
    device: str | torch.device = "cpu",
    use_ema: bool = True,
) -> dict:
    """Load both video DiT and action head from a joint training checkpoint.

    Args:
        video_processor: A ``VideoProcessor`` instance.
        action_model: A ``VideoVLAPytorch`` instance, or None to skip.
        ckpt_dir: Path to a specific step directory (e.g. ``checkpoints/.../14000``).
        device: Device to load weights onto.
        use_ema: If True, prefer EMA action weights.

    Returns:
        Dict with metadata: ``{"step", "video_params_loaded", "action_params_loaded"}``.
    """
    ckpt_dir = Path(ckpt_dir)
    result = {"step": int(ckpt_dir.name) if ckpt_dir.name.isdigit() else -1}

    # Video DiT
    result["video_params_loaded"] = load_video_checkpoint(video_processor, ckpt_dir, device)

    # Action head
    if action_model is not None and (ckpt_dir / "action_model.safetensors").exists():
        result["action_params_loaded"] = load_action_checkpoint(
            action_model, ckpt_dir, device, use_ema=use_ema,
        )
    else:
        result["action_params_loaded"] = 0

    # Metadata
    meta_path = ckpt_dir / "metadata.pt"
    if meta_path.exists():
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        result["step"] = meta.get("global_step", result["step"])

    log.info(f"Loaded joint checkpoint from {ckpt_dir}: step={result['step']}, "
             f"video={result['video_params_loaded']} params, action={result['action_params_loaded']} params")
    return result
