"""
Joint training of Cosmos Video DiT + Action Head (DiT4DiT-style).

Optimized training loop with:
  - EMA (Exponential Moving Average) for both video DiT and action head
  - Mixed precision (bfloat16 autocast)
  - Gradient accumulation
  - Gradient NaN/Inf detection and skipping
  - Separate cosine LR schedules with warmup ratio
  - Video generation callback with wandb video logging
  - Proper checkpoint save/resume for all state (model, EMA, optimizer, step)
  - FSDP (Fully Sharded Data Parallel) support for large models (14B+)

Fully self-contained: no imports from openpi or cosmos_predict2.

Usage:
  torchrun --standalone --nnodes=1 --nproc_per_node=4 \\
      -m temporal_ratio.training.train videoVLA_libero_short

  # With FSDP for 14B model:
  torchrun --standalone --nnodes=1 --nproc_per_node=8 \\
      -m temporal_ratio.training.train videoVLA_libero_short \\
      --fsdp --fsdp_sharding_strategy FULL_SHARD
"""

import argparse
import copy
import dataclasses
import gc
import logging
import math
import glob
import os
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

from temporal_ratio.training import resume as _resume
from temporal_ratio.training.joint_config import JointTrainConfig, VideoDiTConfig
from temporal_ratio.models.video_action_model import JointVideoActionModel
from temporal_ratio.models.video_model import VideoProcessor, IMAGE_SIZES, FEATURE_DIMS, resolve_video_checkpoint
from temporal_ratio.data.joint_loader import create_data_loader, DataConfig
from temporal_ratio.models.action_head import VideoVLAPytorch
from temporal_ratio.models.config import VideoVLAConfig


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------

class EMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of parameters updated as:
        shadow = decay * shadow + (1 - decay) * param
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module, iteration: int = None):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: torch.nn.Module):
        """Replace model params with EMA shadow (for eval/checkpoint)."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module):
        """Restore original params after apply_shadow."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state):
        self.shadow = state["shadow"]
        self.decay = state.get("decay", self.decay)


class PowerEMA:
    """Cosmos-style Power EMA with iteration-dependent decay.

    beta(t) = (1 - 1/(t+1))^(c+1) where c is derived from rate.
    Matches the original Cosmos Predict2.5 EMA implementation.
    Early training: aggressive averaging (beta near 0).
    Late training: slow averaging (beta near 1).
    """

    def __init__(self, model: torch.nn.Module, rate: float = 0.10, iteration_shift: int = 0):
        self.rate = rate
        self.iteration_shift = iteration_shift
        # Derive exponent from rate (matches Cosmos formula)
        self.exp_coeff = float(np.roots([1, 7, 16 - rate**-2, 12 - rate**-2]).real.max())
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def _beta(self, iteration: int) -> float:
        it = iteration + self.iteration_shift
        if it < 1:
            return 0.0
        return (1 - 1 / (it + 1)) ** (self.exp_coeff + 1)

    @torch.no_grad()
    def update(self, model: torch.nn.Module, iteration: int = None):
        if iteration is None:
            iteration = 0
        b = self._beta(iteration)
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(b).add_(param.data, alpha=1.0 - b)

    def apply_shadow(self, model: torch.nn.Module):
        """Replace model params with EMA shadow (for eval/checkpoint)."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module):
        """Restore original params after apply_shadow."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {"shadow": self.shadow, "rate": self.rate, "exp_coeff": self.exp_coeff}

    def load_state_dict(self, state):
        self.shadow = state["shadow"]
        self.rate = state.get("rate", self.rate)
        self.exp_coeff = state.get("exp_coeff", self.exp_coeff)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def log_memory(device, step, phase=""):
    if not torch.cuda.is_available():
        return
    a = torch.cuda.memory_allocated(device) / 1e9
    r = torch.cuda.memory_reserved(device) / 1e9
    p = torch.cuda.max_memory_allocated(device) / 1e9
    logging.info(f"[mem] step={step} {phase}: alloc={a:.2f}GB reserved={r:.2f}GB peak={p:.2f}GB")


def init_wandb(config, ckpt_dir, *, resuming, enabled=True):
    """Start (or resume) a W&B run. The entity comes from ``$WANDB_ENTITY`` / the W&B login."""
    if not enabled:
        wandb.init(mode="disabled")
        return
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    id_file = ckpt_dir / "wandb_id.txt"
    if resuming and id_file.exists():
        run_id = id_file.read_text().strip()
        try:
            wandb.init(id=run_id, resume="must", project=config.project_name)
            return
        except Exception:
            logging.warning(f"Could not resume wandb run {run_id}, starting a new run")
            id_file.unlink(missing_ok=True)
    wandb.init(name=config.exp_name, config=dataclasses.asdict(config), project=config.project_name)
    id_file.write_text(wandb.run.id)


def cosine_lr(step, warmup, total, peak_lr, min_lr):
    """Cosine LR with linear warmup (matches openpi JAX trainer warmup)."""
    if step < warmup:
        # Match JAX behavior: start from peak_lr / (warmup + 1), not 0
        init_lr = peak_lr / (warmup + 1)
        return init_lr + (peak_lr - init_lr) * step / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def cosmos_lr(step, warmup, cycle_length, base_lr, f_max=1.0, f_min=0.2):
    """Cosmos LambdaLinear schedule: warmup to f_max*lr, cosine decay to f_min*lr.

    Matches the original Cosmos Predict2.5 LoRA training schedule.
    """
    if step < warmup:
        return base_lr * f_max * step / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, cycle_length - warmup))
    f = f_min + (f_max - f_min) * 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * f


# ---------------------------------------------------------------------------
# FSDP helpers
# ---------------------------------------------------------------------------

def setup_fsdp_for_dit(video_processor, sharding_strategy_name="FULL_SHARD", use_lora=False):
    """Wrap the DiT network (video_processor.model.net) with FSDP.

    Wraps each transformer block as its own FSDP unit, which enables
    per-block parameter sharding and gradient checkpointing interop.

    Args:
        video_processor: VideoProcessor instance whose .model.net is the DiT.
        sharding_strategy_name: One of FULL_SHARD, SHARD_GRAD_OP, NO_SHARD.
        use_lora: Whether LoRA/PEFT is active on the net (requires use_orig_params=True).

    Returns:
        The FSDP-wrapped net module.
    """
    from functools import partial
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

    cosmos_model = video_processor.model
    net = cosmos_model.net

    # Locate the transformer blocks for the auto-wrap policy.
    # With PEFT/LoRA, blocks live under base_model.model.blocks.
    if hasattr(net, "base_model"):
        blocks = list(net.base_model.model.blocks)
    else:
        blocks = list(getattr(net, "blocks", []))

    if not blocks:
        logging.warning("FSDP: Could not find DiT blocks — wrapping entire net as single FSDP unit")

    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )

    sharding_strategy = ShardingStrategy[sharding_strategy_name]

    # Wrap policy: each transformer block becomes its own FSDP unit
    wrap_policy = partial(
        lambda_auto_wrap_policy,
        lambda_fn=lambda m: m in blocks,
    )

    wrapped_net = FSDP(
        net,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp,
        device_id=torch.cuda.current_device(),
        sharding_strategy=sharding_strategy,
        sync_module_states=True,
        use_orig_params=True,  # Required for PEFT/LoRA compatibility and named_parameters
    )

    # Patch the net back into the model hierarchy
    cosmos_model.net = wrapped_net
    video_processor.compiled_denoiser.model = cosmos_model

    logging.info(
        f"FSDP: Wrapped DiT net with strategy={sharding_strategy_name}, "
        f"{len(blocks)} blocks as FSDP units, use_orig_params=True, "
        f"use_lora={use_lora}"
    )
    return wrapped_net


def fsdp_gather_full_state_dict(fsdp_module):
    """Gather full (unsharded) state dict from an FSDP module on all ranks.

    Uses the FSDP full_state_dict_type context manager so rank 0 gets the
    complete state dict while other ranks get an empty dict.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(fsdp_module, StateDictType.FULL_STATE_DICT, cfg):
        state_dict = fsdp_module.state_dict()
    return state_dict


def fsdp_load_full_state_dict(fsdp_module, state_dict):
    """Load a full (unsharded) state dict into an FSDP module.

    All ranks must call this. Rank 0 provides the full state dict,
    other ranks provide an empty dict. FSDP scatters the parameters.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(fsdp_module, StateDictType.FULL_STATE_DICT, cfg):
        fsdp_module.load_state_dict(state_dict)


def is_fsdp_wrapped(module):
    """Check if a module is wrapped with FSDP."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        return isinstance(module, FSDP)
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Checkpoint save/load (saves EMA too)
# ---------------------------------------------------------------------------

def save_checkpoint(joint_model, optimizer, ema, global_step, config, is_main, ckpt_dir,
                    data_config=None, force=False, fsdp_enabled=False, keep_every=0, extra_metadata=None):
    """Save checkpoint. Called by training loop at save_interval or when force=True.

    Multiples of ``keep_every`` are always saved and never deleted by the cleanup of old
    checkpoints. ``extra_metadata`` is stored in ``metadata.pt`` (e.g. the learning-rate cycle).

    With FSDP enabled, all ranks must participate in the save (for state_dict gathering),
    but only rank 0 writes files to disk.
    """
    force = force or _resume.is_kept(global_step, keep_every)
    # Early exit check: must happen on all ranks when FSDP is disabled
    if not fsdp_enabled:
        if not is_main:
            return
        if not force and config is not None:
            if not (global_step % config.save_interval == 0 and global_step > 0):
                return
    else:
        # With FSDP, all ranks must participate in the gather, but we still
        # need to check whether we should save at this step.
        should_save = force or (config is not None and global_step % config.save_interval == 0 and global_step > 0)
        if not should_save:
            return

    # ── Gather FSDP state dict (all ranks participate) ──
    fsdp_net_state = None
    cosmos_model = joint_model.video_processor.model
    net = cosmos_model.net
    if fsdp_enabled and is_fsdp_wrapped(net):
        logging.info(f"FSDP: Gathering full state dict for checkpoint (step {global_step})...")
        fsdp_net_state = fsdp_gather_full_state_dict(net)
        # After gather, only rank 0 has the state dict; other ranks get empty dict
        if dist.is_initialized():
            dist.barrier()

    # Only rank 0 writes files
    if not is_main:
        return

    final_dir = Path(ckpt_dir) / f"{global_step}"
    tmp_dir = Path(ckpt_dir) / f"tmp_{global_step}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Save action head as model.safetensors (compatible with create_trained_policy)
    action_model = joint_model.action_model
    if isinstance(action_model, torch.nn.parallel.DistributedDataParallel):
        action_model = action_model.module
    safetensors.torch.save_model(action_model, tmp_dir / "model.safetensors")
    # Also save with explicit name for clarity
    safetensors.torch.save_model(action_model, tmp_dir / "action_model.safetensors")

    # Save EMA version of action head
    if ema is not None:
        ema.apply_shadow(joint_model)
        action_model_ema = joint_model.action_model
        if isinstance(action_model_ema, torch.nn.parallel.DistributedDataParallel):
            action_model_ema = action_model_ema.module
        safetensors.torch.save_model(action_model_ema, tmp_dir / "action_model_ema.safetensors")
        ema.restore(joint_model)

    # Save video model checkpoint
    is_peft_model = hasattr(cosmos_model, "net") and hasattr(cosmos_model.net, "peft_config")
    # With FSDP, PEFT detection needs to look through the FSDP wrapper
    if fsdp_enabled and is_fsdp_wrapped(net):
        inner = net.module if hasattr(net, "module") else net
        is_peft_model = hasattr(inner, "peft_config")

    if fsdp_enabled and fsdp_net_state is not None:
        # FSDP mode: save from the gathered full state dict
        full_state = {}
        for name, param in fsdp_net_state.items():
            # Strip FSDP internal prefixes
            clean = name.replace("_fsdp_wrapped_module.", "")
            clean = clean.replace("_checkpoint_wrapped_module.", "")
            if is_peft_model:
                if "lora_A" in clean or "lora_B" in clean:
                    continue
                clean = clean.replace("base_model.model.", "").replace(".base_layer", "")
            full_state[f"net.{clean}"] = param
        torch.save(full_state, tmp_dir / "video_model_full.pt")
        logging.info(f"Saved FSDP-gathered video model ({len(full_state)} keys)")

        # Also save LoRA adapters if PEFT is active
        if is_peft_model:
            try:
                lora_state = {}
                for name, param in fsdp_net_state.items():
                    if "lora_A" in name or "lora_B" in name:
                        lora_state[name] = param
                if lora_state:
                    lora_dir = tmp_dir / "lora_adapters"
                    lora_dir.mkdir(exist_ok=True)
                    torch.save(lora_state, lora_dir / "adapter_model.bin")
                    logging.info(f"Saved LoRA adapters ({len(lora_state)} tensors) to {lora_dir}")
            except Exception as e:
                logging.warning(f"Could not save LoRA adapters under FSDP: {e}")
    elif is_peft_model:
        # Non-FSDP LoRA mode: save adapters (lightweight) for resume,
        # and merged full weights for standalone inference
        try:
            lora_dir = tmp_dir / "lora_adapters"
            lora_dir.mkdir(exist_ok=True)
            cosmos_model.net.save_pretrained(str(lora_dir))
            logging.info(f"Saved LoRA adapters to {lora_dir}")
        except Exception as e:
            logging.warning(f"Could not save LoRA adapters: {e}")

        # Save merged full checkpoint (base + LoRA merged) for inference
        # Must merge LoRA into base weights first, then strip PEFT prefixes
        try:
            from peft import PeftModel
            # Merge LoRA weights into base model (in-place)
            cosmos_model.net.merge_adapter()
            full_state = {}
            for name, param in cosmos_model.net.state_dict().items():
                # Skip LoRA-only params (saved separately above)
                if "lora_A" in name or "lora_B" in name:
                    continue
                # Strip PEFT wrappers: base_model.model. and .base_layer
                clean = name.replace("base_model.model.", "").replace(".base_layer", "")
                clean = clean.replace("_checkpoint_wrapped_module.", "")
                full_state[f"net.{clean}"] = param.data
            # Include _extra_state keys from fused ops (FusedRoPE, FusedAttn)
            # These are static config tensors not exposed by PEFT's state_dict()
            # but required for correct inference when loading standalone
            for name, module in cosmos_model.net.named_modules():
                if hasattr(module, '_extra_state') and callable(getattr(module, 'get_extra_state', None)):
                    try:
                        extra = module.get_extra_state()
                        clean = name.replace("base_model.model.", "").replace(".base_layer", "")
                        clean = clean.replace("_checkpoint_wrapped_module.", "")
                        full_state[f"net.{clean}._extra_state"] = extra
                    except Exception:
                        pass
            torch.save(full_state, tmp_dir / "video_model_full.pt")
            # Unmerge to restore separate LoRA params for continued training
            cosmos_model.net.unmerge_adapter()
        except Exception as e:
            logging.warning(f"Could not save merged video checkpoint: {e}")
            # Try to unmerge in case merge succeeded but save failed
            try:
                cosmos_model.net.unmerge_adapter()
            except Exception:
                pass
    elif hasattr(cosmos_model, "net"):
        # Non-LoRA, non-FSDP: save full state dict as before
        full_state = {}
        for name, param in cosmos_model.net.state_dict().items():
            clean_name = name.replace("_checkpoint_wrapped_module.", "")
            full_state[f"net.{clean_name}"] = param.data
        torch.save(full_state, tmp_dir / "video_model_full.pt")

    # Save REPA projector weights
    if hasattr(joint_model, 'repa_projectors') and joint_model.repa_projectors is not None:
        torch.save(joint_model.repa_projectors.state_dict(), tmp_dir / "repa_projectors.pt")

    # Save EMA state
    if ema is not None:
        torch.save(ema.state_dict(), tmp_dir / "ema.pt")

    # Save optimizer state dict
    # With FSDP + use_orig_params=True, optimizer state is local and can be saved directly.
    # For full resume portability, we save it as-is. On resume the same FSDP config is required.
    torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")

    # Save metadata in the same format as train_videoVLA_flat.py
    metadata = {
        "global_step": global_step,
        "config": dataclasses.asdict(config) if config is not None else {},
        "timestamp": time.time(),
        "fsdp_enabled": fsdp_enabled,
        **(extra_metadata or {}),
    }
    torch.save(metadata, tmp_dir / "metadata.pt")

    # Save norm stats (matching videoVLA checkpoint format)
    if data_config is not None:
        try:
            import temporal_ratio.data.normalize as _normalize
            norm_stats = data_config.norm_stats
            if norm_stats is not None and data_config.asset_id is not None:
                _normalize.save(tmp_dir / "assets" / data_config.asset_id, norm_stats)
                logging.info(f"Saved norm stats to assets/{data_config.asset_id}")
        except Exception as e:
            logging.warning(f"Could not save norm stats: {e}")

    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)
    logging.info(f"Saved checkpoint at step {global_step} -> {final_dir}")

    # Cleanup old checkpoints (keep last N)
    from temporal_ratio.training.rank_monitor import cleanup_old_checkpoints, async_sync_checkpoint, KEEP_LAST_N
    cleanup_old_checkpoints(ckpt_dir, keep_n=KEEP_LAST_N, keep_every=keep_every)

    # Optional background upload: $CHECKPOINT_SYNC_URI/<checkpoint path relative to the run>.
    sync_uri = os.environ.get("CHECKPOINT_SYNC_URI")
    if sync_uri:
        async_sync_checkpoint(str(final_dir.resolve()), f"{sync_uri.rstrip('/')}/{final_dir}")


def load_checkpoint(joint_model, optimizer, ema, ckpt_dir, device, fsdp_enabled=False):
    """Load checkpoint, with FSDP support.

    With FSDP, the video DiT checkpoint (video_model_full.pt) is loaded as a full
    state dict on rank 0 and scattered to all ranks via FSDP.set_state_dict_type.
    All ranks must call this function.
    """
    ckpt_dir = Path(ckpt_dir)
    steps = [int(d.name) for d in ckpt_dir.iterdir()
             if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")]
    if not steps:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    latest = max(steps)
    d = ckpt_dir / f"{latest}"

    is_main = (not dist.is_initialized()) or (dist.get_rank() == 0)

    # Action head (all ranks load independently — it's not FSDP-wrapped)
    p = d / "action_model.safetensors"
    if p.exists():
        m = joint_model.action_model
        if isinstance(m, torch.nn.parallel.DistributedDataParallel):
            m = m.module
        safetensors.torch.load_model(m, p, device=str(device))

    # Video DiT — prefer LoRA adapters, then video_model_full.pt, then legacy
    lora_dir = d / "lora_adapters"
    p_full = d / "video_model_full.pt"
    p_legacy = d / "video_dit_blocks.pt"
    cosmos_net = getattr(getattr(joint_model.video_processor, "model", None), "net", None)

    # Detect PEFT (look through FSDP wrapper if needed)
    is_peft = False
    if cosmos_net is not None:
        if is_fsdp_wrapped(cosmos_net):
            inner = cosmos_net.module if hasattr(cosmos_net, "module") else cosmos_net
            is_peft = hasattr(inner, "peft_config")
        else:
            is_peft = hasattr(cosmos_net, "peft_config")

    if fsdp_enabled and is_fsdp_wrapped(cosmos_net) and p_full.exists():
        # FSDP resume: load full state dict and scatter via FSDP
        logging.info(f"FSDP: Loading video checkpoint from {p_full}")
        if is_main:
            full_state = torch.load(p_full, map_location="cpu", weights_only=True)
            # Strip "net." prefix to get the net-level state dict
            net_state = {}
            for k, v in full_state.items():
                if k.startswith("net."):
                    net_state[k[4:]] = v
                else:
                    net_state[k] = v
        else:
            net_state = {}

        try:
            fsdp_load_full_state_dict(cosmos_net, net_state)
            logging.info(f"FSDP: Loaded video model state dict ({len(net_state)} keys on rank 0)")
        except Exception as e:
            logging.warning(f"FSDP: Failed to load video model state dict: {e}")
    elif lora_dir.exists() and is_peft:
        # Resume LoRA training: load adapter weights into existing PEFT model
        # Base weights are already loaded from the original Cosmos checkpoint
        try:
            from peft import set_peft_model_state_dict
            adapter_path = lora_dir / "adapter_model.safetensors"
            if not adapter_path.exists():
                adapter_path = lora_dir / "adapter_model.bin"
            if adapter_path.exists():
                if str(adapter_path).endswith(".safetensors"):
                    lora_state = safetensors.torch.load_file(str(adapter_path), device=str(device))
                else:
                    lora_state = torch.load(adapter_path, map_location=device, weights_only=True)
                if is_fsdp_wrapped(cosmos_net):
                    # With FSDP + LoRA, load via FSDP full state dict mechanism
                    fsdp_load_full_state_dict(cosmos_net, lora_state)
                else:
                    set_peft_model_state_dict(cosmos_net, lora_state)
                logging.info(f"Loaded LoRA adapters from {lora_dir} ({len(lora_state)} tensors)")
            else:
                logging.warning(f"LoRA dir exists but no adapter weights found in {lora_dir}")
        except Exception as e:
            logging.warning(f"Failed to load LoRA adapters: {e}")
    elif p_full.exists() and not is_peft:
        from temporal_ratio.training.checkpoint_utils import load_video_checkpoint
        load_video_checkpoint(joint_model.video_processor, d, device=device)
    elif p_legacy.exists():
        st = torch.load(p_legacy, map_location=device, weights_only=True)
        joint_model.video_processor.load_state_dict(st, strict=False)
        logging.info(f"Loaded {len(st)} video DiT block params (legacy format)")

    # REPA projectors
    p = d / "repa_projectors.pt"
    if p.exists() and hasattr(joint_model, 'repa_projectors') and joint_model.repa_projectors is not None:
        st = torch.load(p, map_location=device, weights_only=True)
        joint_model.repa_projectors.load_state_dict(st)
        logging.info(f"Loaded REPA projector weights")

    # EMA
    p = d / "ema.pt"
    if p.exists() and ema is not None:
        ema.load_state_dict(torch.load(p, map_location=device, weights_only=False))

    # Optimizer
    p = d / "optimizer.pt"
    if p.exists():
        opt_state = torch.load(p, map_location=device, weights_only=False)
        if len(opt_state["param_groups"]) == len(optimizer.param_groups):
            optimizer.load_state_dict(opt_state)
        else:
            logging.warning(
                f"Optimizer param_groups mismatch: saved={len(opt_state['param_groups'])}, "
                f"current={len(optimizer.param_groups)}. Skipping optimizer restore (LR will reset)."
            )

    meta = torch.load(d / "metadata.pt", map_location=device, weights_only=False)
    return meta.get("global_step", latest)


def latest_checkpoint_metadata(ckpt_dir) -> dict:
    """``metadata.pt`` of the latest checkpoint of a run (empty if the run has none)."""
    steps = _resume.checkpoint_steps(ckpt_dir)
    if not steps:
        return {}
    return torch.load(Path(ckpt_dir) / str(steps[-1]) / "metadata.pt", map_location="cpu", weights_only=False)


def lr_cycle(args, global_step, ckpt_dir, resuming):
    """``(cycle start, warmup steps)`` of the learning-rate cycle; see ``resume.lr_cycle_start``."""
    start = _resume.lr_cycle_start(
        global_step, args.max_train_steps, latest_checkpoint_metadata(ckpt_dir) if resuming else None
    )
    warmup = max(1, int(args.warmup_ratio * (args.max_train_steps - start)))
    if start:
        logging.info(f"Learning-rate cycle from step {start}: {warmup} warmup steps, decay until step "
                     f"{args.max_train_steps}")
    return start, warmup


# ---------------------------------------------------------------------------
# Save demo videos from dataset at start of training
# ---------------------------------------------------------------------------

def save_dataset_demos(loader, ckpt_dir, num_demos=4):
    """Save a few sample videos from the dataset as mp4s for reference."""
    demo_dir = os.path.join(str(ckpt_dir), "dataset_demos")
    os.makedirs(demo_dir, exist_ok=True)

    saved = 0
    for observation, actions, prompts in loader:
        if saved >= num_demos:
            break
        # observation.wan_images_latents: (B, T, H, W, C) tensor
        imgs = observation.wan_images_latents  # (B, T, H, W, C)
        if isinstance(imgs, torch.Tensor):
            imgs = imgs.cpu().numpy()

        B = imgs.shape[0]
        for b in range(min(B, num_demos - saved)):
            frames = imgs[b]  # (T, H, W, C)
            # Normalize to uint8 if needed
            if frames.dtype != np.uint8:
                if frames.max() <= 1.0:
                    frames = (frames * 255).clip(0, 255).astype(np.uint8)
                else:
                    frames = frames.clip(0, 255).astype(np.uint8)

            prompt = prompts[b] if b < len(prompts) else "unknown"
            safe_prompt = "".join(c if c.isalnum() or c in " _-" else "" for c in prompt)[:60]
            vid_path = os.path.join(demo_dir, f"demo_{saved:02d}_{safe_prompt}.mp4")

            try:
                import imageio.v3 as iio
                iio.imwrite(vid_path, frames, fps=10, codec="libx264",
                            plugin="pyav")
            except ImportError:
                try:
                    import cv2
                    h, w = frames.shape[1], frames.shape[2]
                    writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
                    for frame in frames:
                        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    writer.release()
                except Exception as e:
                    logging.warning(f"Failed to save demo video {saved}: {e}")
                    continue

            logging.info(f"Saved dataset demo: {vid_path} ({frames.shape})")
            saved += 1

        if saved >= num_demos:
            break

    logging.info(f"Saved {saved} dataset demo videos to {demo_dir}")

    # Also log to wandb if available
    try:
        if wandb.run is not None:
            demo_files = sorted(glob.glob(os.path.join(demo_dir, "*.mp4")))
            for f in demo_files[:num_demos]:
                wandb.log({"dataset_demo": wandb.Video(f, fps=10, caption=os.path.basename(f))}, step=0)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Video generation callback
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_and_log_videos(joint_model, sample_latent, sample_condition, device,
                            global_step, num_denoising_steps=35, num_autoregressive=10,
                            ckpt_dir=None, rank=0, context_embedding=None,
                            prompt_text=None, first_frame_image=None):
    """Generate autoregressive video and log to disk + wandb.

    Generates multiple 17-frame chunks, using the last frame of each chunk
    as the first (conditioning) frame of the next chunk.
    """
    t0 = time.time()
    joint_model.action_model.eval()
    processor = joint_model.video_processor
    num_frames_per_chunk = 17

    logging.info(
        f"[VideoGen] rank={rank} step={global_step}: "
        f"{num_denoising_steps} denoise x {num_autoregressive} AR chunks"
    )

    try:
        # Get first frame
        if first_frame_image is not None:
            frame = first_frame_image[0]  # (T, H, W, C) or (H, W, C)
            if isinstance(frame, torch.Tensor):
                frame = frame.to(device)
            if frame.dim() == 4:
                frame = frame[0]  # first frame: (H, W, C)
        else:
            frame = torch.randint(0, 256, (224, 448, 3), dtype=torch.uint8, device=device)

        all_decoded_frames = [frame.cpu().numpy().astype(np.uint8)]  # start with conditioning frame
        current_frame = frame  # (H, W, C) uint8

        for ar_step in range(num_autoregressive):
            # Build input: repeat current frame to fill chunk
            video_input = current_frame.unsqueeze(0).unsqueeze(0).repeat(1, num_frames_per_chunk, 1, 1, 1)

            # Generate one chunk
            _, _, final_latent = processor.eval_vid2world(
                images=video_input,
                prompt=prompt_text or None,
                num_conditional_frames=1,
                guidance=7.0,
                seed=42 + rank + ar_step,
                num_steps=num_denoising_steps,
            )

            # Decode latent → pixel frames
            if processor.model.tokenizer is not None:
                decoded = processor.model.decode(final_latent).float()
                chunk_video = decoded.squeeze(0).permute(1, 2, 3, 0)  # (T, H, W, C)
                chunk_video = ((chunk_video + 1) / 2).clamp(0, 1) * 255
                chunk_np = chunk_video.cpu().numpy().astype(np.uint8)
            else:
                chunk_np = video_input[0].cpu().numpy()

            # Append all frames except the first (which is the conditioning frame)
            all_decoded_frames.extend([chunk_np[i] for i in range(1, len(chunk_np))])

            # Use last decoded frame as conditioning for next chunk
            current_frame = torch.from_numpy(chunk_np[-1]).to(device)

            logging.info(f"[VideoGen] rank={rank} AR chunk {ar_step+1}/{num_autoregressive} done")

        full_video = np.stack(all_decoded_frames, axis=0)
        logging.info(f"[VideoGen] rank={rank} Total: {full_video.shape[0]} frames")

    except Exception as e:
        logging.warning(f"[VideoGen] rank={rank} Generation failed: {e}")
        import traceback; traceback.print_exc()
        joint_model.action_model.train()
        return

    joint_model.action_model.train()
    elapsed = time.time() - t0
    logging.info(f"[VideoGen] rank={rank} Time: {elapsed:.1f}s")

    # Save to disk
    save_dir = str(ckpt_dir) if ckpt_dir else "checkpoints"
    vid_dir = os.path.join(save_dir, "generated_videos")
    os.makedirs(vid_dir, exist_ok=True)
    vid_path = os.path.join(vid_dir, f"step_{global_step}_rank{rank}.mp4")
    try:
        import imageio.v3 as iio
        iio.imwrite(vid_path, full_video, fps=10, codec="libx264", plugin="pyav")
        logging.info(f"Saved video to {vid_path}")
    except Exception as e:
        logging.warning(f"Video save failed: {e}")

    # Log to wandb
    if rank == 0 and os.path.exists(vid_path):
        try:
            if wandb.run is not None:
                wandb.log({
                    "generated_video": wandb.Video(vid_path, fps=10, format="mp4",
                                                   caption=f"step_{global_step}_ar{num_autoregressive}"),
                }, step=global_step)
        except Exception as e:
            logging.warning(f"wandb video log failed: {e}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

from temporal_ratio.training.train_args import build_arg_parser, parse_args  # noqa: E402,F401


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_loop(args):
    init_logging()
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    rank = dist.get_rank() if use_ddp else 0
    world_size = dist.get_world_size() if use_ddp else 1
    set_seed(args.seed, local_rank)

    fsdp_enabled = getattr(args, 'fsdp', False)
    if fsdp_enabled and not use_ddp:
        logging.error("FSDP requires distributed training (torchrun with multiple GPUs). "
                      "Use --nproc_per_node=N with N>1, or remove --fsdp.")
        return
    if fsdp_enabled:
        logging.info(f"[INIT] FSDP enabled: strategy={args.fsdp_sharding_strategy}, "
                     f"world_size={world_size}")

    logging.info(f"[INIT] rank={rank} world={world_size} device={device} fsdp={fsdp_enabled}")

    # Enable TF32 for speed on Ampere+
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # ── Video setup: declared by the named robot/dataset config ──
    import temporal_ratio.config.registry as _registry
    video_setup = _registry.get_config(args.data_config_name).video_setup
    is_causal = args.causal
    img_h, img_w = IMAGE_SIZES[video_setup]
    feature_dim = FEATURE_DIMS[video_setup]
    video_latent_shape = ((args.observation_horizon - 1) // 4 + 1, img_h // 16, img_w // 16)

    # ── Config ──
    warmup_steps = max(1, int(args.warmup_ratio * args.max_train_steps))
    video_dit_config = VideoDiTConfig(
        setup=video_setup, feature_layer=args.feature_layer, feature_dim=feature_dim,
        num_feature_timestep_buckets=args.num_feature_timestep_buckets,
        causal_temporal=is_causal, image_height=img_h, image_width=img_w,
        num_video_frames=args.observation_horizon,
    )
    config = JointTrainConfig(
        video_dit=video_dit_config,
        video_loss_weight=args.video_loss_weight,
        video_lr=args.video_lr, action_lr=args.action_lr,
        max_train_steps=args.max_train_steps, warmup_steps=warmup_steps,
        batch_size=args.batch_size, data_config_name=args.data_config_name,
        name=args.data_config_name,
        exp_name=args.exp_name, checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval, save_interval=args.save_interval,
        wandb_enabled=args.wandb_enabled and not args.no_wandb,
        seed=args.seed, resume=args.resume, overwrite=args.overwrite,
        action_head_weight_path=args.action_head_weight_path,
        project_name=args.project_name,
        weight_decay=args.weight_decay, gradient_clip_norm=args.gradient_clip_norm,
    )

    # ── Checkpoint dir: checkpoints/{data_config_name}/{exp_name} ──
    ckpt_dir = Path(config.checkpoint_dir) / config.data_config_name / config.exp_name
    resuming = False
    if config.resume and ckpt_dir.exists():
        steps = [int(d.name) for d in ckpt_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if steps:
            resuming = True
    elif config.overwrite and ckpt_dir.exists() and is_main:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    elif not config.overwrite and _resume.checkpoint_steps(ckpt_dir):
        raise FileExistsError(f"{ckpt_dir} already has checkpoints; resume the run (--resume) or start it "
                              "over (--overwrite)")
    # A continued run draws new data order and noise (a new run keeps the plain seed).
    resume_step = _resume.checkpoint_steps(ckpt_dir)[-1] if resuming else 0
    if resume_step:
        set_seed(args.seed + resume_step, local_rank)
    if not resuming and is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    if use_ddp:
        dist.barrier()
    if is_main:
        init_wandb(config, ckpt_dir, resuming=resuming, enabled=config.wandb_enabled)

    # ── Data loaders (dual: video + action) ──
    # 1. Action dataloader: provides (observation, actions, prompts) with norm stats
    import temporal_ratio.config.registry as _config
    import temporal_ratio.data.action_loader as _data
    existing_config = _config.get_config(args.data_config_name)
    existing_config = dataclasses.replace(existing_config, batch_size=args.batch_size)
    # Override model action_dim so PadStatesAndActions uses the correct dim
    if args.action_dim != existing_config.model.action_dim:
        existing_config = dataclasses.replace(
            existing_config,
            model=dataclasses.replace(existing_config.model, action_dim=args.action_dim),
        )
    data_cfg = existing_config.data.create(existing_config.assets_dirs, existing_config.model)
    action_loader = _data.create_data_loader(
        dataclasses.replace(existing_config, seed=existing_config.seed + resume_step),
        framework="pytorch", shuffle=True, flat_frames=True,
    )
    logging.info(f"Action dataloader: {args.data_config_name}")

    # 2. Video dataloader: provides {video, ai_caption, ...} for video DiT loss
    from temporal_ratio.data.video_datasets import get_dataset_config
    video_dataset_name = getattr(args, "video_dataset", None)
    video_loader = None
    if video_dataset_name and config.video_loss_weight > 0:
        video_data_config = get_dataset_config(video_dataset_name)
        video_data_config.batch_size = args.batch_size
        video_loader = video_data_config.build_dataloader(seed=resume_step)
        _n_sources = len(video_data_config.datasets)
        logging.info(f"Video dataloader: {video_dataset_name} ({_n_sources} sources)")
    else:
        logging.info("No separate video dataloader — video loss uses action data")

    # ── Build models ──
    state_dim = args.state_dim if args.state_dim is not None else args.action_dim
    # Use action_horizon from data config (source of truth) unless explicitly overridden
    action_horizon = existing_config.model.action_horizon
    if action_horizon != args.action_horizon:
        logging.info(f"Using action_horizon={action_horizon} from data config (CLI default was {args.action_horizon})")
    model_cfg = VideoVLAConfig(dtype="bfloat16", action_dim=args.action_dim, state_dim=state_dim, action_horizon=action_horizon)
    action_model = VideoVLAPytorch(model_cfg, video_latent_shape, video_feature_dim=feature_dim)

    if config.action_head_weight_path:
        p = os.path.join(config.action_head_weight_path, "model.safetensors")
        if os.path.exists(p):
            safetensors.torch.load_model(action_model, p, device="cpu")
            logging.info(f"Loaded action head from {p}")

    # Build video processor
    base_ckpt = resolve_video_checkpoint(args.base_cosmos_ckpt)
    logging.info(f"Using VideoProcessor: setup={video_setup}, ckpt={base_ckpt or 'default'}")
    video_processor = VideoProcessor(
        setup=video_setup,
        device=device,
        atten_temporal_causal=is_causal,
        ckpt_path=base_ckpt,
    )

    # ── LoRA injection ──
    lora_enabled = getattr(args, 'lora_enabled', False)
    if lora_enabled:
        lora_target = args.lora_target_modules
        logging.info(f"Injecting LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}, "
                     f"targets={lora_target}")
        # add_lora is on the V2W model, wraps model.net with PEFT
        cosmos_model = video_processor.model
        cosmos_model.net = cosmos_model.add_lora(
            network=cosmos_model.net,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_target_modules=lora_target,
        )
        # Update compiled denoiser to use the new LoRA-wrapped net
        video_processor.compiled_denoiser.model = cosmos_model

        # Count LoRA vs total params
        lora_p = sum(p.numel() for p in cosmos_model.net.parameters() if p.requires_grad)
        total_p = sum(p.numel() for p in cosmos_model.net.parameters())
        logging.info(f"LoRA: {lora_p:,} trainable / {total_p:,} total DiT params "
                     f"({100*lora_p/total_p:.3f}%)")

        # frozen_video with LoRA: freeze everything (including LoRA params).
        # Use joint_detach explicitly if you want LoRA to train via video loss.

        # Upcast LoRA params to fp32 (matches Cosmos FusedAdamW master weights)
        if getattr(args, 'fp32_lora_params', True):
            n_upcasted = 0
            for name, param in cosmos_model.net.named_parameters():
                if param.requires_grad and "lora_" in name:
                    param.data = param.data.float()
                    n_upcasted += 1
            logging.info(f"Upcasted {n_upcasted} LoRA params to fp32 (matches Cosmos)")

    # Disable conditioner's built-in text dropout during tokenize_video —
    # it was stacking with manual --text_dropout_rate giving ~36% instead of 20%.
    # The conditioner's 0.2 dropout is the correct Cosmos behavior; we just need
    # to ensure it's the ONLY source. Manual dropout code has been removed.
    if args.text_dropout_rate > 0:
        logging.warning(
            f"--text_dropout_rate={args.text_dropout_rate} is DEPRECATED and ignored. "
            f"Text dropout is handled by the conditioner (0.2 from Cosmos config)."
        )

    # REPA config
    repa_config = None
    if args.repa_enabled:
        from temporal_ratio.training.joint_config import REPAConfig
        repa_config = REPAConfig(
            enabled=True,
            encoder_names=args.repa_encoder,
            repa_layers=args.repa_layers,
            proj_coeff=args.repa_coeff,
            projector_dim=args.repa_projector_dim,
            repa_lr=args.repa_lr,
        )
        logging.info(f"REPA enabled: encoders={args.repa_encoder}, layers={args.repa_layers}, "
                     f"coeff={args.repa_coeff}, lr={args.repa_lr}")

    joint_model = JointVideoActionModel(
        action_model=action_model, video_processor=video_processor,
        video_config=video_dit_config,
        video_loss_weight=config.video_loss_weight,
        training_mode=args.training_mode, device=device,
        repa_config=repa_config,
        state_dropout_rate=getattr(args, "state_dropout_rate", 0.0),
        first_frame_drop_prob=getattr(args, "action_first_frame_drop_prob", 0.0),
    )
    if joint_model.first_frame_drop_prob > 0:
        logging.info(f"Action head: first latent frame hidden with p={joint_model.first_frame_drop_prob}")

    # Enable gradient checkpointing on both video DiT and action head
    # (unless --no_gradient_checkpointing is set, which trades memory for speed)
    if not getattr(args, "no_gradient_checkpointing", False):
        if hasattr(joint_model.action_model, "gradient_checkpointing_enable"):
            joint_model.action_model.gradient_checkpointing_enable()
        # Also enable on video DiT blocks if supported
        model_net = getattr(joint_model.video_processor, "model", None)
        if model_net and hasattr(model_net, "net"):
            # With PEFT/LoRA, blocks live under base_model.model.blocks
            dit_net = model_net.net
            if hasattr(dit_net, "base_model"):
                blocks = getattr(dit_net.base_model.model, "blocks", [])
            else:
                blocks = getattr(dit_net, "blocks", [])
            for block in blocks:
                if hasattr(block, "gradient_checkpointing"):
                    block.gradient_checkpointing = True
    else:
        # Unwrap CheckpointWrapper modules baked into Cosmos model at load time
        def _get_dit_blocks_owner(vp):
            net = vp.model.net
            if hasattr(net, "base_model") and hasattr(net.base_model, "model"):
                return net.base_model.model
            return net
        blocks_owner = _get_dit_blocks_owner(joint_model.video_processor)
        count = 0
        for block_id, block in blocks_owner.blocks.named_children():
            if hasattr(block, "_checkpoint_wrapped_module"):
                blocks_owner.blocks.register_module(block_id, block._checkpoint_wrapped_module)
                count += 1
        if hasattr(blocks_owner, "final_layer") and hasattr(blocks_owner.final_layer, "_checkpoint_wrapped_module"):
            blocks_owner.final_layer = blocks_owner.final_layer._checkpoint_wrapped_module
            count += 1
        logging.info(f"Gradient checkpointing DISABLED: unwrapped {count} CheckpointWrapper modules")

    # ── FSDP wrapping ──
    # Must happen after LoRA injection and gradient checkpointing, before optimizer creation.
    # FSDP wraps the DiT net; other components (action head, REPA, VAE) are small enough
    # to remain as-is (DDP or replicated).
    if fsdp_enabled:
        setup_fsdp_for_dit(
            joint_model.video_processor,
            sharding_strategy_name=args.fsdp_sharding_strategy,
            use_lora=lora_enabled,
        )
        if is_main:
            log_memory(device, 0, "after_fsdp_wrapping")

    # Keep a reference for attribute access (same object — no DDP wrapping).
    joint_model_unwrapped = joint_model

    if use_ddp:
        dist.barrier()
    if is_main:
        log_memory(device, 0, "after_model_creation")

    # ── EMA ──
    # EMA is incompatible with FSDP because FSDP shards parameters across ranks,
    # so named_parameters() returns flat shards, not full tensors. Maintaining a
    # shadow copy per-shard is possible but error-prone; for now we disable EMA
    # when FSDP is active and rely on the FSDP-gathered full checkpoint for eval.
    if fsdp_enabled and (args.ema_decay > 0 or args.ema_type == "power"):
        logging.warning("EMA is disabled when FSDP is enabled (parameter sharding is incompatible "
                        "with EMA shadow copies). Use the saved full checkpoint for evaluation.")
        ema = None
    elif args.ema_type == "power":
        ema = PowerEMA(joint_model_unwrapped, rate=args.ema_rate)
        logging.info(f"Power EMA enabled: rate={args.ema_rate}, exp_coeff={ema.exp_coeff:.3f}")
    else:
        ema = EMA(joint_model_unwrapped, decay=args.ema_decay) if args.ema_decay > 0 else None
        if ema:
            logging.info(f"Constant EMA enabled with decay={args.ema_decay}")

    # ── Optimizer (separate LRs, proper weight decay) ──
    # Don't apply weight decay to biases and LayerNorm.
    # Note: With FSDP (use_orig_params=True), named_parameters() returns
    # original param names/tensors, so this works identically to non-FSDP.
    video_params_decay, video_params_no_decay = [], []
    action_params_decay, action_params_no_decay = [], []

    for name, param in joint_model_unwrapped.video_processor.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in name or "norm" in name.lower():
            video_params_no_decay.append(param)
        else:
            video_params_decay.append(param)

    for name, param in joint_model_unwrapped.action_model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in name or "norm" in name.lower():
            action_params_no_decay.append(param)
        else:
            action_params_decay.append(param)

    # REPA projector params
    repa_params_decay, repa_params_no_decay = [], []
    if hasattr(joint_model, 'repa_projectors') and joint_model_unwrapped.repa_projectors is not None:
        for name, param in joint_model_unwrapped.repa_projectors.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or "bias" in name:
                repa_params_no_decay.append(param)
            else:
                repa_params_decay.append(param)

    repa_lr = args.repa_lr if args.repa_enabled else args.action_lr
    # When LoRA is enabled, video_params contain only LoRA weights;
    # use lora_lr if specified, otherwise video_lr
    effective_video_lr = args.video_lr
    if lora_enabled and args.lora_lr is not None:
        effective_video_lr = args.lora_lr
        logging.info(f"Using LoRA-specific LR: {effective_video_lr}")

    param_groups = [
        {"params": video_params_decay, "lr": effective_video_lr, "weight_decay": args.weight_decay, "name": "video_dit"},
        {"params": video_params_no_decay, "lr": effective_video_lr, "weight_decay": 0.0, "name": "video_dit_no_wd"},
        {"params": action_params_decay, "lr": args.action_lr, "weight_decay": args.weight_decay, "name": "action_head"},
        {"params": action_params_no_decay, "lr": args.action_lr, "weight_decay": 0.0, "name": "action_head_no_wd"},
        {"params": repa_params_decay, "lr": repa_lr, "weight_decay": args.weight_decay, "name": "repa_proj"},
        {"params": repa_params_no_decay, "lr": repa_lr, "weight_decay": 0.0, "name": "repa_proj_no_wd"},
    ]
    # Filter empty groups
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, args.beta2), eps=1e-8)

    # ── Resume ──
    global_step = 0
    if resuming:
        global_step = load_checkpoint(joint_model, optimizer, ema, str(ckpt_dir), device,
                                      fsdp_enabled=fsdp_enabled)
        logging.info(f"Resumed from step {global_step}")
    lr_cycle_start, warmup_steps = lr_cycle(args, global_step, ckpt_dir, resuming)

    # ── Autocast context ──
    use_amp = torch.cuda.is_available() and device.type == "cuda"
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    # ── Training loop ──
    joint_model_unwrapped.action_model.train()
    start_time = time.time()
    infos = []
    accum_steps = args.gradient_accumulation_steps
    nan_count = 0

    if is_main:
        total_p = sum(p.numel() for p in joint_model.parameters())
        train_p = sum(p.numel() for p in joint_model.parameters() if p.requires_grad)
        logging.info(f"Params: total={total_p:,} trainable={train_p:,}")
        if fsdp_enabled:
            logging.info(f"FSDP: strategy={args.fsdp_sharding_strategy}, "
                         f"EMA={'disabled (FSDP)' if ema is None else f'decay={args.ema_decay}'}")
        if lora_enabled:
            logging.info(f"LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}, LR={effective_video_lr:.2e}")
        logging.info(f"Video LR={effective_video_lr:.2e} Action LR={args.action_lr:.2e} lambda={args.video_loss_weight}")
        logging.info(f"Text dropout: conditioner built-in 0.2 (Cosmos default)")
        logging.info(f"Optimizer: AdamW betas=(0.9, {args.beta2}), WD={args.weight_decay}, LR schedule={args.lr_schedule}")
        if args.lr_schedule == "cosmos":
            logging.info(f"Cosmos LR: f_max={args.lr_f_max}, f_min={args.lr_f_min}")
        logging.info(f"Warmup={warmup_steps} Accum={accum_steps} EMA={args.ema_type}(rate={args.ema_rate if args.ema_type=='power' else args.ema_decay})")
        if args.mixed_conditional_frames:
            logging.info(f"Mixed conditional frames: probs={args.conditional_frames_probs}")
        eff_bs = args.batch_size * world_size * accum_steps
        logging.info(f"Effective batch size: {args.batch_size} x {world_size} GPUs x {accum_steps} accum = {eff_bs}")

    pbar = tqdm.tqdm(total=args.max_train_steps, initial=global_step, desc="Training",
                     disable=not is_main) if is_main else None

    if use_ddp:
        dist.barrier()

    # Save dataset demo videos before training starts
    if is_main and global_step == 0:
        save_dataset_demos(action_loader, ckpt_dir, num_demos=4)

    # Pre-encode all video dataset prompts if video_loader exists
    if video_loader is not None:
        logging.info("Video dataloader provided — dual dataloader mode")

    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    video_iter = None

    def _next_video_batch():
        """Get next batch from video dataloader (infinite cycling)."""
        nonlocal video_iter
        if video_loader is None:
            return None
        for _ in range(3):
            if video_iter is None:
                video_iter = iter(video_loader)
            try:
                return next(video_iter)
            except StopIteration:
                video_iter = None
        raise RuntimeError("Video dataloader empty after 3 re-iteration attempts")

    while global_step < args.max_train_steps:
        for observation, actions, prompts in action_loader:
            if global_step >= args.max_train_steps:
                break

            observation = _to_device(observation, device)
            actions = actions.to(torch.float32).to(device)

            # Tokenize ACTION data (frozen VAE + Reason1 text encoding)
            # Text dropout is handled by the conditioner (0.2 from Cosmos config)
            with torch.no_grad():
                action_prompt = joint_model_unwrapped.video_processor.encode_prompt(prompts)

                # Mixed conditional frames: sample from {0,1,2}
                ncf = 1  # default: video2world
                if args.mixed_conditional_frames:
                    _cf_probs = [float(x) for x in args.conditional_frames_probs.split(",")]
                    ncf = int(np.random.choice(len(_cf_probs), p=_cf_probs))

                action_latent, action_condition = joint_model_unwrapped.tokenize_video(
                    observation.wan_images_latents, action_prompt,
                    num_conditional_frames=ncf,
                )

            # Tokenize VIDEO data if separate dataloader exists (skip in frozen_video)
            video_latent, video_condition = None, None
            if video_loader is not None and args.training_mode != "frozen_video":
                video_batch = _next_video_batch()
                if video_batch is not None:
                    with torch.no_grad():
                        # video_batch["video"] is (B, C, T, H, W) uint8
                        vid_frames = video_batch["video"].to(device)
                        vid_frames = vid_frames.permute(0, 2, 3, 4, 1)  # → (B, T, H, W, C)
                        vid_prompt = video_batch["ai_caption"]
                        if isinstance(vid_prompt, (list, tuple)):
                            vid_prompt = list(vid_prompt)
                        vid_prompt_emb = joint_model_unwrapped.video_processor.encode_prompt(vid_prompt)

                        video_latent, video_condition = joint_model_unwrapped.tokenize_video(
                            vid_frames, vid_prompt_emb,
                        )

            # Step-0 video generation callback (skip under FSDP — dtype mismatch)
            if global_step == 0 and args.video_log_interval > 0 and micro_step == 0 and not fsdp_enabled:
                if use_ddp:
                    dist.barrier()
                generate_and_log_videos(
                    joint_model_unwrapped, action_latent, action_condition, device, 0,
                    args.video_gen_steps, args.video_autoregressive_steps,
                    ckpt_dir=ckpt_dir, rank=rank,
                    context_embedding=action_prompt,
                    prompt_text=prompts[0] if prompts else "",
                    first_frame_image=observation.wan_images_latents,
                )
                if use_ddp:
                    dist.barrier()

            # Forward with dual data
            with amp_ctx:
                losses = joint_model(
                    observation=observation, actions=actions,
                    action_latent=action_latent, action_condition=action_condition,
                    video_latent=video_latent, video_condition=video_condition,
                )
                loss = losses["total_loss"] / accum_steps

            # Backward
            loss.backward()
            micro_step += 1

            # Only step optimizer after accumulation
            if micro_step % accum_steps != 0:
                continue

            # All-reduce gradients across ranks.
            # - FSDP mode: FSDP syncs DiT params; manually sync non-FSDP params.
            # - Non-FSDP multi-GPU: manually sync ALL trainable params.
            if use_ddp:
                if fsdp_enabled and is_fsdp_wrapped(joint_model_unwrapped.video_processor.model.net):
                    fsdp_net = joint_model_unwrapped.video_processor.model.net
                    fsdp_param_ids = {id(p) for p in fsdp_net.parameters()}
                    for p in joint_model.parameters():
                        if p.requires_grad and p.grad is not None and id(p) not in fsdp_param_ids:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                else:
                    for p in joint_model.parameters():
                        if p.requires_grad and p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

            # Check for NaN/Inf gradients and clip
            # With FSDP, use FSDP.clip_grad_norm_ for correct cross-rank gradient norm
            if fsdp_enabled and is_fsdp_wrapped(joint_model_unwrapped.video_processor.model.net):
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                fsdp_net = joint_model_unwrapped.video_processor.model.net
                # Clip FSDP-wrapped DiT params via FSDP's method
                grad_norm_fsdp = fsdp_net.clip_grad_norm_(args.gradient_clip_norm)
                # Clip non-FSDP params (action head, REPA) normally
                fsdp_param_ids = {id(p) for p in fsdp_net.parameters()}
                non_fsdp_params = [p for p in joint_model.parameters()
                                   if p.requires_grad and p.grad is not None
                                   and id(p) not in fsdp_param_ids]
                if non_fsdp_params:
                    grad_norm_other = torch.nn.utils.clip_grad_norm_(non_fsdp_params, args.gradient_clip_norm)
                    grad_norm = (grad_norm_fsdp ** 2 + grad_norm_other ** 2) ** 0.5
                else:
                    grad_norm = grad_norm_fsdp
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in joint_model.parameters() if p.requires_grad],
                    max_norm=args.gradient_clip_norm,
                )

            if not torch.isfinite(grad_norm):
                nan_count += 1
                logging.warning(f"NaN/Inf grad at step {global_step} (count={nan_count}), skipping")
                optimizer.zero_grad(set_to_none=True)
                if nan_count > 50:
                    logging.error("Too many NaN gradients, stopping")
                    return
                continue

            # Update LR per group
            for pg in optimizer.param_groups:
                name = pg.get("name", "")
                if "video" in name:
                    base_lr = effective_video_lr
                elif "repa" in name:
                    base_lr = args.repa_lr if args.repa_enabled else args.action_lr
                else:
                    base_lr = args.action_lr
                if args.lr_schedule == "cosmos":
                    pg["lr"] = cosmos_lr(global_step - lr_cycle_start, warmup_steps,
                                         args.max_train_steps - lr_cycle_start, base_lr, args.lr_f_max, args.lr_f_min)
                else:
                    min_lr = base_lr * args.min_lr_ratio
                    pg["lr"] = cosine_lr(global_step - lr_cycle_start, warmup_steps,
                                         args.max_train_steps - lr_cycle_start, base_lr, min_lr)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Broadcast parameters from rank 0 to keep all GPUs in sync
            # (all-reduce on gradients alone is insufficient if optimizer states diverge)
            if use_ddp and not fsdp_enabled:
                for p in joint_model.parameters():
                    if p.requires_grad:
                        dist.broadcast(p.data, src=0)

            # EMA update
            if ema is not None:
                ema.update(joint_model, iteration=global_step)

            # Logging
            if is_main:
                info = {
                    "total_loss": losses["total_loss"].item(),
                    "video_loss": losses["video_loss"].item(),
                    "action_loss": losses["action_loss"].item(),
                    "grad_norm": float(grad_norm),
                    "video_lr": optimizer.param_groups[0]["lr"],
                    "action_lr": optimizer.param_groups[2]["lr"] if len(optimizer.param_groups) > 2 else optimizer.param_groups[-1]["lr"],
                }
                if "repa_loss" in losses:
                    info["repa_loss"] = losses["repa_loss"].item()
                infos.append(info)

            if is_main and global_step % config.log_interval == 0 and infos:
                elapsed = time.time() - start_time
                avg = {k: sum(d[k] for d in infos) / len(infos) for k in infos[0]}
                steps_per_sec = len(infos) / max(elapsed, 1e-6)
                repa_str = f" repa={avg['repa_loss']:.4f}" if 'repa_loss' in avg else ""
                logging.info(
                    f"step={global_step} total={avg['total_loss']:.4f} "
                    f"vid={avg['video_loss']:.4f} act={avg['action_loss']:.4f}{repa_str} "
                    f"gnorm={avg['grad_norm']:.2f} vlr={avg['video_lr']:.2e} alr={avg['action_lr']:.2e} "
                    f"{steps_per_sec:.1f} steps/s"
                )
                if config.wandb_enabled:
                    wandb.log({**avg, "steps_per_sec": steps_per_sec, "nan_grad_count": nan_count}, step=global_step)
                start_time = time.time()
                infos = []

            if is_main and global_step < 5:
                log_memory(device, global_step, "after_backward")

            # Periodic normalized L2 measurement vs GT actions (every 100 steps, rank 0 only;
            # not in video_only mode, where the action head is not run)
            if is_main and global_step % 100 == 0 and hasattr(joint_model_unwrapped, "_last_features"):
                try:
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        action_head = joint_model_unwrapped.action_model
                        was_training = action_head.training
                        action_head.train(False)
                        pred = action_head.sample_actions(
                            device, observation,
                            joint_model_unwrapped._last_features,
                            joint_model_unwrapped._last_tau_f,
                        )
                        action_head.train(was_training)
                        gt = actions[:, :pred.shape[1], :14].to(pred.dtype)
                        pred_14 = pred[:, :, :14]
                        # Both pred and gt are in normalized (z-score) space
                        # L2 in this space = normalized L2
                        norm_l2 = (pred_14 - gt).pow(2).mean(dim=-1).sqrt().mean().item()
                        norm_mse = (pred_14 - gt).pow(2).mean().item()
                        logging.info(f"[gt_l2] step={global_step} norm_l2={norm_l2:.3f} norm_mse={norm_mse:.4f}")
                        if config.wandb_enabled:
                            wandb.log({"gt_norm_l2": norm_l2, "gt_norm_mse": norm_mse}, step=global_step)
                except Exception as e:
                    logging.warning(f"[gt_l2] step={global_step} failed: {e}")

            global_step += 1

            # Save checkpoint first, then run callbacks
            # With FSDP, all ranks must participate in save_checkpoint for state dict gathering
            save_checkpoint(joint_model_unwrapped, optimizer, ema, global_step, config, is_main, str(ckpt_dir),
                           data_config=data_cfg if 'data_cfg' in dir() else None,
                           fsdp_enabled=fsdp_enabled, keep_every=args.keep_every,
                           extra_metadata={"lr_cycle_start": lr_cycle_start})

            # Video generation callback (skip under FSDP — dtype mismatch with mixed precision)
            if (args.video_log_interval > 0 and global_step % args.video_log_interval == 0
                    and not fsdp_enabled):
                if use_ddp:
                    dist.barrier()
                generate_and_log_videos(
                    joint_model_unwrapped, action_latent, action_condition, device, global_step,
                    args.video_gen_steps, args.video_autoregressive_steps,
                    ckpt_dir=ckpt_dir, rank=rank,
                    context_embedding=action_prompt,
                    prompt_text=prompts[0] if prompts else "",
                    first_frame_image=observation.wan_images_latents,
                )
                if use_ddp:
                    dist.barrier()

            if pbar:
                pbar.update(1)
                pbar.set_postfix(loss=f"{losses['total_loss'].item():.4f}")

    if pbar:
        pbar.close()

    # ── Final checkpoint (if not already saved at this step) ──
    if global_step % config.save_interval != 0 and not _resume.is_kept(global_step, args.keep_every):
        save_checkpoint(
            joint_model_unwrapped, optimizer, ema, global_step, config,
            is_main, str(ckpt_dir),
            data_config=data_cfg if "data_cfg" in dir() else None,
            fsdp_enabled=fsdp_enabled, force=True, keep_every=args.keep_every,
            extra_metadata={"lr_cycle_start": lr_cycle_start},
        )
        if is_main:
            logging.info(f"Final checkpoint saved at step {global_step}")

    if is_main and config.wandb_enabled:
        wandb.finish()
    cleanup_ddp()


def _to_device(observation, device):
    import dataclasses as dc
    updates = {}
    for f in dc.fields(observation):
        val = getattr(observation, f.name)
        if isinstance(val, torch.Tensor):
            updates[f.name] = val.to(device, non_blocking=True)
        else:
            updates[f.name] = val
    return dc.replace(observation, **updates)


def main():
    args = parse_args()
    train_loop(args)


if __name__ == "__main__":
    main()
