"""
Dual-resolution joint training: long-horizon video loss + short-horizon action loss.

Key design choices:
  - Video DiT trains on N-frame sequences (default 93) for richer temporal dynamics.
    93 frames → 24 temporal latent tokens → 9408 DiT tokens (LIBERO 224×448).
  - Action head trains on 17-frame features (standard action horizon).
  - Sequential backward passes within each step:
      Phase 1: 93f video forward on video data → video loss → backward (frees 93f activations)
      Phase 2: 17f video forward on video data → video loss → backward (frees 17f activations)
      Phase 3: 17f feature extraction on action data → action loss → backward
    Peak memory = max(phase1, phase2, phase3) instead of sum.
  - Supports LoRA and full fine-tuning (with FSDP).
  - Multi-node training via torchrun.

Usage:
  # LoRA, 2 nodes × 8 GPUs:
  torchrun --nnodes=2 --nproc_per_node=8 \\
      --master_addr=$MASTER_ADDR --master_port=29500 --node_rank=$NODE_RANK \\
      -m temporal_ratio.training.train_dual_resolution videoVLA_libero_short \\
      --lora_enabled --lora_rank 32 \\
      --video_batch_size 2 --action_batch_size 4

  # Full fine-tuning + FSDP, 2 nodes × 8 GPUs:
  torchrun --nnodes=2 --nproc_per_node=8 \\
      --master_addr=$MASTER_ADDR --master_port=29500 --node_rank=$NODE_RANK \\
      -m temporal_ratio.training.train_dual_resolution videoVLA_libero_short \\
      --fsdp --fsdp_sharding_strategy FULL_SHARD \\
      --video_batch_size 1 --action_batch_size 4
"""

import os

import argparse
import dataclasses
import gc
import logging
import math
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
import tqdm
import wandb

# Re-use all utilities from the 17-frame training script
from temporal_ratio.training.train import EMA, PowerEMA, _to_device, cleanup_ddp, cosine_lr, cosmos_lr, generate_and_log_videos, init_logging, init_wandb, is_fsdp_wrapped, load_checkpoint, log_memory, lr_cycle, save_checkpoint, set_seed, setup_ddp, setup_fsdp_for_dit
from temporal_ratio.training import resume as _resume
from temporal_ratio.training.joint_config import JointTrainConfig, VideoDiTConfig
from temporal_ratio.models.video_action_model import JointVideoActionModel
from temporal_ratio.models.video_model import VideoProcessor, FEATURE_DIMS, IMAGE_SIZES, resolve_video_checkpoint
from temporal_ratio.models.action_head import VideoVLAPytorch
from temporal_ratio.models.config import VideoVLAConfig


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

from temporal_ratio.training.train_dual_resolution_args import build_arg_parser, parse_args  # noqa: E402,F401


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

    fsdp_enabled = getattr(args, "fsdp", False)
    if fsdp_enabled and not use_ddp:
        logging.error("FSDP requires distributed training (torchrun). Remove --fsdp or add GPUs.")
        return

    logging.info(
        f"[INIT] rank={rank} world={world_size} device={device} fsdp={fsdp_enabled} "
        f"video_frames={args.video_num_frames} video_bs={args.video_batch_size} "
        f"action_bs={args.action_batch_size}"
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # ── Video setup: declared by the named robot/dataset config ──
    import temporal_ratio.config.registry as _registry
    video_setup = _registry.get_config(args.data_config_name).video_setup
    is_causal = args.causal
    img_h, img_w = IMAGE_SIZES[video_setup]
    feature_dim = FEATURE_DIMS[video_setup]
    # Action head uses 17-frame latent shape (unchanged)
    action_latent_shape = (
        (args.observation_horizon - 1) // 4 + 1,
        img_h // 16,
        img_w // 16,
    )
    # Video loss uses N-frame latent shape (for logging only — the model handles it dynamically)
    video_temporal_latent = (args.video_num_frames - 1) // 4 + 1
    video_tokens = video_temporal_latent * (img_h // 16) * (img_w // 16)
    action_tokens = action_latent_shape[0] * action_latent_shape[1] * action_latent_shape[2]

    logging.info(
        f"Video latent: T'={video_temporal_latent}, tokens={video_tokens} "
        f"({args.video_num_frames} frames)"
    )
    logging.info(
        f"Action latent: T'={action_latent_shape[0]}, tokens={action_tokens} "
        f"({args.observation_horizon} frames)"
    )

    # ── Config ──
    warmup_steps = max(1, int(args.warmup_ratio * args.max_train_steps))
    # VideoDiTConfig uses 17-frame settings (for action head)
    video_dit_config = VideoDiTConfig(
        setup=video_setup, feature_layer=args.feature_layer, feature_dim=feature_dim,
        num_feature_timestep_buckets=args.num_feature_timestep_buckets,
        causal_temporal=is_causal, image_height=img_h, image_width=img_w,
        num_video_frames=args.observation_horizon,  # 17 — for action head
    )
    config = JointTrainConfig(
        video_dit=video_dit_config,
        video_loss_weight=args.video_loss_weight,
        video_lr=args.video_lr, action_lr=args.action_lr,
        max_train_steps=args.max_train_steps, warmup_steps=warmup_steps,
        batch_size=args.action_batch_size, data_config_name=args.data_config_name,
        name=args.data_config_name,
        exp_name=args.exp_name, checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval, save_interval=args.save_interval,
        wandb_enabled=args.wandb_enabled and not args.no_wandb,
        seed=args.seed, resume=args.resume, overwrite=args.overwrite,
        action_head_weight_path=args.action_head_weight_path,
        project_name=args.project_name,
        weight_decay=args.weight_decay, gradient_clip_norm=args.gradient_clip_norm,
    )

    # ── Checkpoint dir ──
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

    # ══════════════════════════════════════════════════════════════════════
    # DATA LOADERS
    # ══════════════════════════════════════════════════════════════════════

    # 1. Action dataloader: 17-frame, yields (Observation, actions, prompts)
    import temporal_ratio.config.registry as _config
    import temporal_ratio.data.action_loader as _data

    existing_config = _config.get_config(args.data_config_name)
    existing_config = dataclasses.replace(
        existing_config, batch_size=args.action_batch_size, seed=existing_config.seed + resume_step
    )
    if args.action_dim != existing_config.model.action_dim:
        existing_config = dataclasses.replace(
            existing_config,
            model=dataclasses.replace(existing_config.model, action_dim=args.action_dim),
        )
    data_cfg = existing_config.data.create(existing_config.assets_dirs, existing_config.model)
    action_loader = _data.create_data_loader(
        existing_config, framework="pytorch", shuffle=True, flat_frames=True
    )
    logging.info(f"Action dataloader: {args.data_config_name} (17 frames, BS={args.action_batch_size})")

    # Val episode IDs for nL2 callback (held-out 20%, loaded lazily)
    val_episode_ids = None
    val_loader = None
    val_iter = None
    if args.val_split > 0:
        import lerobot.common.datasets.lerobot_dataset as _lerobot
        _all_repo_ids = [existing_config.data.repo_id]
        if hasattr(existing_config.data, "extra_repo_ids"):
            _all_repo_ids += list(existing_config.data.extra_repo_ids)
        rng_val = np.random.RandomState(args.val_seed)
        val_episode_ids = {}
        for rid in _all_repo_ids:
            meta = _lerobot.LeRobotDatasetMetadata(rid)
            n_eps = meta.total_episodes
            n_val = max(1, int(n_eps * args.val_split))
            all_eps = list(range(n_eps))
            rng_val.shuffle(all_eps)
            val_episode_ids[rid] = sorted(all_eps[:n_val])
            if is_main:
                logging.info(f"  val: {rid}: {n_val}/{n_eps} eps held out")

    # 2. Video dataloader: N-frame (default 93), yields {video, ai_caption, ...}
    from temporal_ratio.data.video_datasets import get_dataset_config
    from temporal_ratio.data.video_loader import create_dataloader as _create_vdl, create_weighted_dataloader as _create_wvdl
    from torch.utils.data import ConcatDataset as _ConcatDataset

    video_data_config = get_dataset_config(
        args.video_dataset,
        num_frames=args.video_num_frames,
        batch_size=args.video_batch_size,
        frame_skip=args.video_frame_skip,
    )
    # Build with optimized dataloader settings
    _vdl_kwargs = {"seed": resume_step}
    _vdl_nw = getattr(args, "video_num_workers", 8)
    if getattr(args, "video_prefetch_factor", None) is not None:
        _vdl_kwargs["prefetch_factor"] = args.video_prefetch_factor
    if getattr(args, "persistent_workers", False) and _vdl_nw > 0:
        _vdl_kwargs["persistent_workers"] = True
    _vdl_built = [d.build() for d in video_data_config.datasets]
    if len(_vdl_built) == 1:
        video_loader = _create_vdl(_vdl_built[0], batch_size=args.video_batch_size, num_workers=_vdl_nw, **_vdl_kwargs)
    elif video_data_config.weights is not None:
        video_loader = _create_wvdl(_vdl_built, video_data_config.weights, batch_size=args.video_batch_size, num_workers=_vdl_nw, **_vdl_kwargs)
    else:
        video_loader = _create_vdl(_ConcatDataset(_vdl_built), batch_size=args.video_batch_size, num_workers=_vdl_nw, **_vdl_kwargs)
    _n_sources = len(video_data_config.datasets)
    logging.info(
        f"Video dataloader (93f): {args.video_dataset} ({args.video_num_frames} frames, "
        f"BS={args.video_batch_size}, workers={_vdl_nw}, {_n_sources} sources, "
        f"prefetch={_vdl_kwargs.get('prefetch_factor', 2)}, persistent={_vdl_kwargs.get('persistent_workers', False)})"
    )

    # 3. Video dataloader: 17-frame (same dataset, different frame count)
    video_data_config_17f = get_dataset_config(
        args.video_dataset,
        num_frames=17,
        batch_size=args.video_batch_size,
        frame_skip=args.video_frame_skip,
    )
    _vdl_built_17f = [d.build() for d in video_data_config_17f.datasets]
    if len(_vdl_built_17f) == 1:
        video_loader_17f = _create_vdl(_vdl_built_17f[0], batch_size=args.video_batch_size, num_workers=_vdl_nw, **_vdl_kwargs)
    elif video_data_config_17f.weights is not None:
        video_loader_17f = _create_wvdl(_vdl_built_17f, video_data_config_17f.weights, batch_size=args.video_batch_size, num_workers=_vdl_nw, **_vdl_kwargs)
    else:
        video_loader_17f = _create_vdl(_ConcatDataset(_vdl_built_17f), batch_size=args.video_batch_size, num_workers=_vdl_nw, **_vdl_kwargs)
    logging.info(
        f"Video dataloader (17f): {args.video_dataset} (17 frames, "
        f"BS={args.video_batch_size}, workers={_vdl_nw}, {_n_sources} sources)"
    )

    # ══════════════════════════════════════════════════════════════════════
    # MODELS
    # ══════════════════════════════════════════════════════════════════════

    state_dim = args.state_dim if args.state_dim is not None else args.action_dim
    # Use action_horizon from data config (source of truth) unless explicitly overridden
    action_horizon = existing_config.model.action_horizon
    if action_horizon != args.action_horizon:
        logging.info(f"Using action_horizon={action_horizon} from data config (CLI default was {args.action_horizon})")
    model_cfg = VideoVLAConfig(
        dtype="bfloat16", action_dim=args.action_dim, state_dim=state_dim,
        action_horizon=action_horizon,
    )
    action_model = VideoVLAPytorch(model_cfg, action_latent_shape, video_feature_dim=feature_dim)

    if config.action_head_weight_path:
        p = os.path.join(config.action_head_weight_path, "model.safetensors")
        if os.path.exists(p):
            safetensors.torch.load_model(action_model, p, device="cpu")
            logging.info(f"Loaded action head from {p}")

    # Build video processor
    base_ckpt = resolve_video_checkpoint(args.base_cosmos_ckpt)
    logging.info(f"VideoProcessor: setup={video_setup}, ckpt={base_ckpt or 'default'}")
    video_processor = VideoProcessor(
        setup=video_setup, device=device,
        atten_temporal_causal=is_causal, ckpt_path=base_ckpt,
    )

    # LoRA injection
    lora_enabled = getattr(args, "lora_enabled", False)
    if lora_enabled:
        _inject_lora(args, video_processor)
        # Upcast LoRA params to fp32 (matches Cosmos FusedAdamW master_weights)
        if getattr(args, "fp32_lora_params", True):
            cosmos_model = video_processor.model
            fp32_count = 0
            for name, param in cosmos_model.net.named_parameters():
                if param.requires_grad and "lora_" in name:
                    param.data = param.data.float()
                    fp32_count += 1
            logging.info(f"Upcasted {fp32_count} LoRA params to fp32")

    # REPA config
    repa_config = None
    if args.repa_enabled:
        from temporal_ratio.training.joint_config import REPAConfig
        repa_config = REPAConfig(
            enabled=True, encoder_names=args.repa_encoder, repa_layers=args.repa_layers,
            proj_coeff=args.repa_coeff, projector_dim=args.repa_projector_dim,
            repa_lr=args.repa_lr,
        )

    # JointVideoActionModel uses 17-frame config (action head shape)
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

    # Speed optimizations
    if getattr(args, "no_gradient_checkpointing", False):
        _disable_dit_gradient_checkpointing(video_processor)
    elif getattr(args, "sac_mode", None) is not None:
        _set_dit_sac_mode(video_processor, args.sac_mode)
    if getattr(args, "compile", False):
        _compile_dit_blocks(video_processor)

    # FSDP wrapping (after LoRA + grad ckpt, before optimizer)
    if fsdp_enabled:
        setup_fsdp_for_dit(
            joint_model.video_processor,
            sharding_strategy_name=args.fsdp_sharding_strategy,
            use_lora=lora_enabled,
        )
        if is_main:
            log_memory(device, 0, "after_fsdp_wrapping")

    joint_model_unwrapped = joint_model

    if use_ddp:
        dist.barrier()
    if is_main:
        log_memory(device, 0, "after_model_creation")

    # ── EMA (disabled with FSDP) ──
    if fsdp_enabled and args.ema_decay > 0:
        logging.warning("EMA disabled under FSDP.")
        ema = None
    elif getattr(args, "ema_type", "constant") == "power":
        ema = PowerEMA(joint_model_unwrapped, rate=args.ema_rate)
        logging.info(f"Using PowerEMA (rate={args.ema_rate})")
    else:
        ema = EMA(joint_model_unwrapped, decay=args.ema_decay) if args.ema_decay > 0 else None

    # ══════════════════════════════════════════════════════════════════════
    # OPTIMIZER
    # ══════════════════════════════════════════════════════════════════════

    effective_video_lr = args.video_lr
    if lora_enabled and args.lora_lr is not None:
        effective_video_lr = args.lora_lr

    param_groups = _build_param_groups(
        joint_model_unwrapped, effective_video_lr, args.action_lr,
        args.repa_lr if args.repa_enabled else args.action_lr,
        args.weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, args.beta2), eps=1e-8)

    # ── Resume ──
    global_step = 0
    if resuming:
        global_step = load_checkpoint(
            joint_model, optimizer, ema, str(ckpt_dir), device,
            fsdp_enabled=fsdp_enabled,
        )
        logging.info(f"Resumed from step {global_step}")
    lr_cycle_start, warmup_steps = lr_cycle(args, global_step, ckpt_dir, resuming)

    # ── Autocast ──
    use_amp = torch.cuda.is_available() and device.type == "cuda"
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    # ══════════════════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ══════════════════════════════════════════════════════════════════════

    joint_model_unwrapped.action_model.train()
    start_time = time.time()
    infos = []
    accum_steps = args.gradient_accumulation_steps
    nan_count = 0
    micro_step = 0
    video_iter = None
    video_iter_17f = None

    proj_coeff = (
        repa_config.proj_coeff
        if (repa_config and repa_config.enabled) else 0.0
    )

    # ── Deprecation warning for text_dropout_rate ──
    if getattr(args, "text_dropout_rate", 0.0) > 0:
        logging.warning(
            "⚠ --text_dropout_rate is DEPRECATED and IGNORED. "
            "The conditioner already applies dropout_rate=0.2 from experiment config. "
            "Remove --text_dropout_rate from your launch script."
        )

    if is_main:
        total_p = sum(p.numel() for p in joint_model.parameters())
        train_p = sum(p.numel() for p in joint_model.parameters() if p.requires_grad)
        logging.info(f"Params: total={total_p:,} trainable={train_p:,}")
        eff_video_bs = args.video_batch_size * world_size * accum_steps
        eff_action_bs = args.action_batch_size * world_size * accum_steps
        logging.info(
            f"Effective batch sizes: video={eff_video_bs} "
            f"({args.video_batch_size}×{world_size}×{accum_steps}) | "
            f"action={eff_action_bs} ({args.action_batch_size}×{world_size}×{accum_steps})"
        )
        logging.info(
            f"Video LR={effective_video_lr:.2e} Action LR={args.action_lr:.2e} "
            f"lambda={args.video_loss_weight} mode={args.training_mode}"
        )
        if args.video_aux_every_k > 1:
            logging.info(f"93f auxiliary video loss runs every {args.video_aux_every_k} steps")
        ema_desc = f"PowerEMA(rate={args.ema_rate})" if getattr(args, "ema_type", "constant") == "power" else f"EMA(decay={args.ema_decay})"
        logging.info(
            f"Warmup={warmup_steps} Accum={accum_steps} {ema_desc} "
            f"WD={args.weight_decay} beta2={args.beta2} lr_schedule={args.lr_schedule}"
        )

    # ── Rank collapse monitor ──
    rank_monitor = None
    if is_main and lora_enabled:
        try:
            from temporal_ratio.training.rank_monitor import RankCollapseMonitor
            rank_monitor = RankCollapseMonitor(
                net=joint_model_unwrapped.video_processor.model.net,
                device=device,
            )
            logging.info("[RankMonitor] Initialized (logs at every checkpoint save)")
        except Exception as e:
            logging.warning(f"[RankMonitor] Could not initialize: {e}")

    pbar = (
        tqdm.tqdm(total=args.max_train_steps, initial=global_step, desc="Training")
        if is_main else None
    )

    if use_ddp:
        dist.barrier()

    def _next_video_batch():
        nonlocal video_iter
        if video_iter is None:
            video_iter = iter(video_loader)
        try:
            return next(video_iter)
        except StopIteration:
            video_iter = iter(video_loader)
            return next(video_iter)

    def _next_video_batch_17f():
        nonlocal video_iter_17f
        if video_iter_17f is None:
            video_iter_17f = iter(video_loader_17f)
        try:
            return next(video_iter_17f)
        except StopIteration:
            video_iter_17f = iter(video_loader_17f)
            return next(video_iter_17f)

    optimizer.zero_grad(set_to_none=True)

    while global_step < args.max_train_steps:
        for observation, actions, prompts in action_loader:
            if global_step >= args.max_train_steps:
                break

            observation = _to_device(observation, device)
            actions = actions.to(torch.float32).to(device)

            # ── Tokenize ACTION data (17 frames, frozen) ──
            with torch.no_grad():
                action_prompt = joint_model_unwrapped.video_processor.encode_prompt(prompts)
                # Note: text dropout handled by conditioner (dropout_rate=0.2 in config)

                # Mixed conditional frames (Cosmos-style multi-mode training)
                action_ncf_kwargs = {}
                if getattr(args, "mixed_conditional_frames", False):
                    cond_probs = [float(x) for x in args.conditional_frames_probs.split(",")]
                    ncf = np.random.choice(len(cond_probs), p=cond_probs)
                    action_ncf_kwargs["num_conditional_frames"] = ncf

                action_latent, action_cond = joint_model_unwrapped.tokenize_video(
                    observation.wan_images_latents, action_prompt,
                    **action_ncf_kwargs,
                )

            # ── Tokenize VIDEO data (N frames, frozen) ──
            # Skip 93f phase when not needed (video_aux_every_k > 1)
            run_93f = (args.video_aux_every_k <= 1) or (global_step % args.video_aux_every_k == 0)

            video_loss_93f = torch.tensor(0.0, device=device)
            repa_loss = torch.tensor(0.0, device=device)

            if run_93f:
                video_batch = _next_video_batch()
                with torch.no_grad():
                    vid_frames = video_batch["video"].to(device)
                    vid_frames = vid_frames.permute(0, 2, 3, 4, 1)  # → (B, T, H, W, C)
                    vid_captions = video_batch["ai_caption"]
                    if isinstance(vid_captions, (list, tuple)):
                        vid_captions = list(vid_captions)
                    vid_prompt = joint_model_unwrapped.video_processor.encode_prompt(vid_captions)
                    # Note: text dropout handled by conditioner (dropout_rate=0.2 in config)

                    # Mixed conditional frames for video data too
                    vid_ncf_kwargs = {}
                    if getattr(args, "mixed_conditional_frames", False):
                        cond_probs = [float(x) for x in args.conditional_frames_probs.split(",")]
                        ncf = np.random.choice(len(cond_probs), p=cond_probs)
                        vid_ncf_kwargs["num_conditional_frames"] = ncf

                    video_latent, video_cond = joint_model_unwrapped.tokenize_video(
                        vid_frames, vid_prompt,
                        **vid_ncf_kwargs,
                    )

            # ══════════════════════════════════════════════════════════════
            # PHASE 1: 93-frame video loss (forward + backward)
            #   93-frame activations are freed after backward()
            #   Skipped when video_aux_every_k > 1 and not on a K-th step
            # ══════════════════════════════════════════════════════════════

            if run_93f and args.training_mode not in ("frozen_video",):
                with amp_ctx:
                    video_loss_93f, repa_loss = joint_model_unwrapped.video_forward(
                        video_latent, video_cond,
                    )
                scaled_93f = (
                    args.video_loss_weight * video_loss_93f + proj_coeff * repa_loss
                ) / accum_steps
                scaled_93f.backward()
                # 93-frame computation graph freed here

            # ══════════════════════════════════════════════════════════════
            # PHASE 2: 17-frame video-only loss on video data
            #   Same dataset as Phase 1 but at 17-frame resolution.
            #   Activations freed after backward().
            # ══════════════════════════════════════════════════════════════

            video_loss_17f_vid = torch.tensor(0.0, device=device)

            if run_93f and args.training_mode not in ("frozen_video",):
                video_batch_17f = _next_video_batch_17f()
                with torch.no_grad():
                    vid_frames_17f = video_batch_17f["video"].to(device)
                    vid_frames_17f = vid_frames_17f.permute(0, 2, 3, 4, 1)
                    vid_captions_17f = video_batch_17f["ai_caption"]
                    if isinstance(vid_captions_17f, (list, tuple)):
                        vid_captions_17f = list(vid_captions_17f)
                    vid_prompt_17f = joint_model_unwrapped.video_processor.encode_prompt(vid_captions_17f)

                    vid_ncf_kwargs_17f = {}
                    if getattr(args, "mixed_conditional_frames", False):
                        cond_probs = [float(x) for x in args.conditional_frames_probs.split(",")]
                        ncf = np.random.choice(len(cond_probs), p=cond_probs)
                        vid_ncf_kwargs_17f["num_conditional_frames"] = ncf

                    video_latent_17f, video_cond_17f = joint_model_unwrapped.tokenize_video(
                        vid_frames_17f, vid_prompt_17f,
                        **vid_ncf_kwargs_17f,
                    )

                with amp_ctx:
                    video_loss_17f_vid, _ = joint_model_unwrapped.video_forward(
                        video_latent_17f, video_cond_17f,
                    )
                scaled_17f_vid = (
                    args.video_loss_weight * video_loss_17f_vid
                ) / accum_steps
                scaled_17f_vid.backward()
                # 17-frame video-only computation graph freed here

            # ══════════════════════════════════════════════════════════════
            # PHASE 3: 17-frame video loss + action loss (single DiT fwd)
            #   video_forward_with_features → video_loss_17f + features
            # ══════════════════════════════════════════════════════════════

            video_loss_17f = torch.tensor(0.0, device=device)
            action_loss = torch.tensor(0.0, device=device)

            if args.training_mode != "video_only":
                with amp_ctx:
                    video_loss_17f, features, tau_f = \
                        joint_model_unwrapped.video_forward_with_features(
                            action_latent, action_cond,
                        )

                import dataclasses as dc
                if hasattr(observation, "state") and observation.state.dtype == torch.float64:
                    observation = dc.replace(observation, state=observation.state.float())
                actions_f = actions.float()

                with amp_ctx:
                    action_losses = joint_model_unwrapped.action_model(
                        observation, actions_f, features, tau_f,
                        first_frame_drop_prob=joint_model_unwrapped.first_frame_drop_prob,
                    )
                action_loss = action_losses.mean()
                scaled_17f = (
                    args.video_loss_weight * video_loss_17f + action_loss
                ) / accum_steps
                scaled_17f.backward()

            # ── Gradient accumulation gate ──
            micro_step += 1
            if micro_step % accum_steps != 0:
                continue

            # ── Gradient sync across ranks ──
            if use_ddp:
                if fsdp_enabled and is_fsdp_wrapped(
                    joint_model_unwrapped.video_processor.model.net
                ):
                    fsdp_net = joint_model_unwrapped.video_processor.model.net
                    fsdp_ids = {id(p) for p in fsdp_net.parameters()}
                    for p in joint_model.parameters():
                        if p.requires_grad and p.grad is not None and id(p) not in fsdp_ids:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                else:
                    for p in joint_model.parameters():
                        if p.requires_grad and p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

            # ── Gradient clipping ──
            if fsdp_enabled and is_fsdp_wrapped(
                joint_model_unwrapped.video_processor.model.net
            ):
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                fsdp_net = joint_model_unwrapped.video_processor.model.net
                grad_norm_fsdp = fsdp_net.clip_grad_norm_(args.gradient_clip_norm)
                fsdp_ids = {id(p) for p in fsdp_net.parameters()}
                non_fsdp = [
                    p for p in joint_model.parameters()
                    if p.requires_grad and p.grad is not None and id(p) not in fsdp_ids
                ]
                if non_fsdp:
                    grad_norm_other = torch.nn.utils.clip_grad_norm_(
                        non_fsdp, args.gradient_clip_norm
                    )
                    grad_norm = (grad_norm_fsdp ** 2 + grad_norm_other ** 2) ** 0.5
                else:
                    grad_norm_other = torch.tensor(0.0)
                    grad_norm = grad_norm_fsdp
                grad_norm_video = grad_norm_fsdp
                grad_norm_action = grad_norm_other
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in joint_model.parameters() if p.requires_grad],
                    max_norm=args.gradient_clip_norm,
                )
                # Compute per-component grad norms (read-only, after clipping)
                video_params = [
                    p for p in joint_model_unwrapped.video_processor.parameters()
                    if p.requires_grad and p.grad is not None
                ]
                action_params = [
                    p for p in joint_model_unwrapped.action_model.parameters()
                    if p.requires_grad and p.grad is not None
                ]
                grad_norm_video = torch.sqrt(sum(p.grad.norm() ** 2 for p in video_params)) if video_params else torch.tensor(0.0)
                grad_norm_action = torch.sqrt(sum(p.grad.norm() ** 2 for p in action_params)) if action_params else torch.tensor(0.0)

            if not torch.isfinite(grad_norm):
                nan_count += 1
                logging.warning(f"NaN/Inf grad at step {global_step} (count={nan_count})")
                optimizer.zero_grad(set_to_none=True)
                if nan_count > 50:
                    logging.error("Too many NaN gradients, stopping")
                    return
                continue

            # ── LR schedule ──
            for pg in optimizer.param_groups:
                name = pg.get("name", "")
                if "video" in name:
                    base_lr = effective_video_lr
                elif "repa" in name:
                    base_lr = args.repa_lr if args.repa_enabled else args.action_lr
                else:
                    base_lr = args.action_lr
                min_lr = base_lr * args.min_lr_ratio
                if getattr(args, "lr_schedule", "cosine") == "cosmos":
                    pg["lr"] = cosmos_lr(
                        global_step - lr_cycle_start, warmup_steps, args.max_train_steps - lr_cycle_start,
                        base_lr, args.lr_f_max, args.lr_f_min,
                    )
                else:
                    pg["lr"] = cosine_lr(
                        global_step - lr_cycle_start, warmup_steps, args.max_train_steps - lr_cycle_start,
                        base_lr, min_lr,
                    )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # ── Broadcast params (non-FSDP multi-GPU) ──
            if use_ddp and not fsdp_enabled:
                for p in joint_model.parameters():
                    if p.requires_grad:
                        dist.broadcast(p.data, src=0)

            if ema is not None:
                ema.update(joint_model, iteration=global_step)

            # ── Logging ──
            if is_main:
                info = {
                    "total_loss": (
                        (video_loss_93f + video_loss_17f_vid + video_loss_17f) * args.video_loss_weight
                        + action_loss
                    ).item(),
                    "video_loss": (video_loss_93f + video_loss_17f_vid + video_loss_17f).item(),
                    "video_loss_93f": video_loss_93f.item(),
                    "video_loss_17f_vid": video_loss_17f_vid.item(),
                    "video_loss_17f": video_loss_17f.item(),
                    "action_loss": action_loss.item(),
                    "grad_norm": float(grad_norm),
                    "grad_norm_video": float(grad_norm_video),
                    "grad_norm_action": float(grad_norm_action),
                    "video_lr": optimizer.param_groups[0]["lr"],
                    "action_lr": (
                        optimizer.param_groups[2]["lr"]
                        if len(optimizer.param_groups) > 2
                        else optimizer.param_groups[-1]["lr"]
                    ),
                }
                if proj_coeff > 0:
                    info["repa_loss"] = repa_loss.item()
                infos.append(info)

            if is_main and global_step % config.log_interval == 0 and infos:
                elapsed = time.time() - start_time
                avg = {k: sum(d[k] for d in infos) / len(infos) for k in infos[0]}
                sps = len(infos) / max(elapsed, 1e-6)
                repa_str = f" repa={avg['repa_loss']:.4f}" if "repa_loss" in avg else ""
                logging.info(
                    f"step={global_step} total={avg['total_loss']:.4f} "
                    f"vid93={avg['video_loss_93f']:.4f} vid17v={avg['video_loss_17f_vid']:.4f} vid17={avg['video_loss_17f']:.4f} "
                    f"act={avg['action_loss']:.4f}{repa_str} "
                    f"gnorm={avg['grad_norm']:.2f} vlr={avg['video_lr']:.2e} "
                    f"alr={avg['action_lr']:.2e} {sps:.1f} steps/s"
                )
                if config.wandb_enabled:
                    wandb.log(
                        {**avg, "steps_per_sec": sps, "nan_grad_count": nan_count},
                        step=global_step,
                    )
                start_time = time.time()
                infos = []

            if is_main and global_step < 5:
                log_memory(device, global_step, "after_backward")

            global_step += 1

            # ── Checkpoint ──
            save_checkpoint(
                joint_model_unwrapped, optimizer, ema, global_step, config,
                is_main, str(ckpt_dir),
                data_config=data_cfg if "data_cfg" in dir() else None,
                fsdp_enabled=fsdp_enabled, keep_every=args.keep_every,
                extra_metadata={"lr_cycle_start": lr_cycle_start},
            )

            # ── Rank collapse monitoring (at checkpoint intervals) ──
            if (is_main and global_step % config.save_interval == 0
                    and global_step > 0 and rank_monitor is not None):
                try:
                    rm_metrics = rank_monitor.compute_metrics(global_step)
                    if config.wandb_enabled:
                        wandb.log(rm_metrics, step=global_step)
                    logging.info(
                        f"[RankMonitor] step={global_step} "
                        f"b27_v={rm_metrics.get('rank_monitor/b27_cross_v_proj_ratio', 0):.3f} "
                        f"b27_out={rm_metrics.get('rank_monitor/b27_cross_output_proj_ratio', 0):.3f} "
                        f"drift_ratio={rm_metrics.get('rank_monitor/avg_drift_ratio', 0):.2f} "
                        f"avg_cross_drift={rm_metrics.get('rank_monitor/avg_cross_drift', 0):.4f} "
                        f"avg_self_drift={rm_metrics.get('rank_monitor/avg_self_drift', 0):.4f}"
                    )
                except Exception as e:
                    logging.warning(f"[RankMonitor] Error: {e}")

            # ── Video generation callback (17-frame, skip under FSDP) ──
            if (
                args.video_log_interval > 0
                and global_step % args.video_log_interval == 0
                and not fsdp_enabled
            ):
                if use_ddp:
                    dist.barrier()
                generate_and_log_videos(
                    joint_model_unwrapped, action_latent, action_cond, device,
                    global_step, args.video_gen_steps, args.video_autoregressive_steps,
                    ckpt_dir=ckpt_dir, rank=rank,
                    context_embedding=action_prompt,
                    prompt_text=prompts[0] if prompts else "",
                    first_frame_image=observation.wan_images_latents,
                )
                if use_ddp:
                    dist.barrier()

            # ── nL2 callback (at save_interval, all ranks) ──
            if (
                config.save_interval > 0
                and global_step % config.save_interval == 0
                and args.training_mode != "video_only"
            ):
                nl2_t = torch.tensor(0.0, device=device)
                nl2_left_t = torch.tensor(0.0, device=device)
                nl2_right_t = torch.tensor(0.0, device=device)
                n_val_batches = 0
                try:
                    action_head = joint_model_unwrapped.action_model
                    was_training = action_head.training
                    action_head.train(False)

                    if val_episode_ids is not None:
                        # Lazily create val loader on first use
                        if val_loader is None:
                            val_config = dataclasses.replace(existing_config, batch_size=min(4, args.action_batch_size))
                            val_loader = _data.create_data_loader(
                                val_config, framework="pytorch", shuffle=False, flat_frames=True,
                                episode_ids_per_repo=val_episode_ids,
                            )
                            val_iter = iter(val_loader)
                        # Use held-out val data
                        _val_nl2s, _val_ls, _val_rs = [], [], []
                        for _vb in range(5):
                            try:
                                val_obs, val_acts, val_prompts = next(val_iter)
                            except StopIteration:
                                val_iter = iter(val_loader)
                                val_obs, val_acts, val_prompts = next(val_iter)
                            val_obs = _to_device(val_obs, device)
                            val_acts = val_acts.to(torch.float32).to(device)
                            val_captions = list(val_prompts) if isinstance(val_prompts, (list, tuple)) else [val_prompts]
                            val_prompt_emb = joint_model_unwrapped.video_processor.encode_prompt(val_captions)
                            with torch.no_grad():
                                val_latent, val_cond = joint_model_unwrapped.tokenize_video(
                                    val_obs.wan_images_latents, val_prompt_emb,
                                )
                            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                                _, val_feats, val_tau = joint_model_unwrapped.video_forward_with_features(val_latent, val_cond)
                                val_pred = action_head.sample_actions(device, val_obs, val_feats.detach(), val_tau.detach())
                            val_pred_f = val_pred.float()
                            val_gt_f = val_acts.float()
                            val_err = val_pred_f[:, :val_gt_f.shape[1], :val_gt_f.shape[2]] - val_gt_f
                            _val_nl2s.append(torch.sqrt((val_err ** 2).mean(dim=-1)).mean())
                            _val_ls.append(torch.sqrt((val_err[..., :7] ** 2).mean(dim=-1)).mean())
                            _val_rs.append(torch.sqrt((val_err[..., 7:] ** 2).mean(dim=-1)).mean())
                        if _val_nl2s:
                            nl2_t = torch.stack(_val_nl2s).mean()
                            nl2_left_t = torch.stack(_val_ls).mean()
                            nl2_right_t = torch.stack(_val_rs).mean()
                            n_val_batches = len(_val_nl2s)
                    elif features is not None:
                        # Fallback: use current training batch
                        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            pred = action_head.sample_actions(device, observation, features.detach(), tau_f.detach())
                        pred_f = pred.float()
                        gt_f = actions.float()
                        err = pred_f[:, :gt_f.shape[1], :gt_f.shape[2]] - gt_f
                        nl2_t = torch.sqrt((err ** 2).mean(dim=-1)).mean()
                        nl2_left_t = torch.sqrt((err[..., :7] ** 2).mean(dim=-1)).mean()
                        nl2_right_t = torch.sqrt((err[..., 7:] ** 2).mean(dim=-1)).mean()
                        n_val_batches = 1

                    action_head.train(was_training)
                except Exception as e:
                    if is_main:
                        logging.warning(f"[nL2] local compute failed at step {global_step}: {e}")
                if use_ddp:
                    dist.barrier()
                    for t in [nl2_t, nl2_left_t, nl2_right_t]:
                        dist.all_reduce(t, op=dist.ReduceOp.AVG)
                if is_main:
                    nl2 = nl2_t.item()
                    nl2_left = nl2_left_t.item()
                    nl2_right = nl2_right_t.item()
                    src = "val" if val_iter is not None else "train"
                    logging.info(f"[nL2:{src}] step={global_step} nL2={nl2:.4f} L={nl2_left:.4f} R={nl2_right:.4f} ({n_val_batches} batches)")
                    if config.wandb_enabled:
                        wandb.log({"eval/nl2": nl2, "eval/nl2_left": nl2_left, "eval/nl2_right": nl2_right}, step=global_step)

            if pbar:
                pbar.update(1)
                pbar.set_postfix(
                    v93=f"{video_loss_93f.item():.3f}",
                    v17v=f"{video_loss_17f_vid.item():.3f}",
                    v17=f"{video_loss_17f.item():.3f}",
                    aloss=f"{action_loss.item():.3f}",
                )

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


# ---------------------------------------------------------------------------
# Helpers (avoid modifying existing code)
# ---------------------------------------------------------------------------

def _inject_lora(args, video_processor):
    """Inject LoRA adapters into the DiT."""
    lora_target = args.lora_target_modules
    logging.info(
        f"Injecting LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}, "
        f"targets={lora_target}"
    )
    cosmos_model = video_processor.model
    cosmos_model.net = cosmos_model.add_lora(
        network=cosmos_model.net,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=lora_target,
    )
    video_processor.compiled_denoiser.model = cosmos_model

    lora_p = sum(p.numel() for p in cosmos_model.net.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in cosmos_model.net.parameters())
    logging.info(
        f"LoRA: {lora_p:,} trainable / {total_p:,} total "
        f"({100 * lora_p / total_p:.3f}%)"
    )


def _enable_gradient_checkpointing(joint_model):
    """Enable gradient checkpointing on action head and DiT blocks."""
    if hasattr(joint_model.action_model, "gradient_checkpointing_enable"):
        joint_model.action_model.gradient_checkpointing_enable()
    model_net = getattr(joint_model.video_processor, "model", None)
    if model_net and hasattr(model_net, "net"):
        dit_net = model_net.net
        if hasattr(dit_net, "base_model"):
            blocks = getattr(dit_net.base_model.model, "blocks", [])
        else:
            blocks = getattr(dit_net, "blocks", [])
        for block in blocks:
            if hasattr(block, "gradient_checkpointing"):
                block.gradient_checkpointing = True
        logging.info(f"Gradient checkpointing enabled on {len(blocks)} DiT blocks")


def _get_dit_blocks_owner(video_processor):
    """Get the actual DiT module that owns .blocks and .final_layer, navigating through PeftModel."""
    net = video_processor.model.net
    if hasattr(net, "base_model") and hasattr(net.base_model, "model"):
        return net.base_model.model
    return net


def _unwrap_checkpoint_wrappers(blocks_owner):
    """Unwrap all CheckpointWrapper modules from DiT blocks and final_layer."""
    count = 0
    for block_id, block in blocks_owner.blocks.named_children():
        if hasattr(block, "_checkpoint_wrapped_module"):
            blocks_owner.blocks.register_module(block_id, block._checkpoint_wrapped_module)
            count += 1
    if hasattr(blocks_owner, "final_layer") and hasattr(blocks_owner.final_layer, "_checkpoint_wrapped_module"):
        blocks_owner.final_layer = blocks_owner.final_layer._checkpoint_wrapped_module
        count += 1
    return count


def _disable_dit_gradient_checkpointing(video_processor):
    """Fully disable gradient checkpointing. WARNING: requires ~2x memory for 93f."""
    blocks_owner = _get_dit_blocks_owner(video_processor)
    count = _unwrap_checkpoint_wrappers(blocks_owner)
    logging.info(f"Disabled gradient checkpointing: unwrapped {count} CheckpointWrapper modules")


def _set_dit_sac_mode(video_processor, mode: str):
    """Switch the SAC policy to a lighter/heavier mode.

    Available modes: none, predict2_2b_720, predict2_2b_720_aggressive, block_wise, mm_only
    The default loaded model uses 'predict2_2b_720_aggressive' which recomputes all mm ops.
    'predict2_2b_720' saves some mm ops for ~10-20% speedup at ~15 GB more memory.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper as ptd_checkpoint_wrapper,
    )
    from temporal_ratio.models.cosmos_dit import SACConfig, CheckpointMode

    blocks_owner = _get_dit_blocks_owner(video_processor)

    # First unwrap existing checkpoint wrappers
    unwrapped = _unwrap_checkpoint_wrappers(blocks_owner)
    logging.info(f"Unwrapped {unwrapped} existing CheckpointWrapper modules")

    if mode == "none":
        logging.info("SAC mode set to NONE — no gradient checkpointing")
        return

    # Re-wrap with new SAC config
    sac_config = SACConfig(mode=mode)
    context_fn = sac_config.get_context_fn()

    count = 0
    for block_id, block in blocks_owner.blocks.named_children():
        wrapped = ptd_checkpoint_wrapper(block, context_fn=context_fn, preserve_rng_state=False)
        blocks_owner.blocks.register_module(block_id, wrapped)
        count += 1

    blocks_owner.final_layer = ptd_checkpoint_wrapper(
        blocks_owner.final_layer, context_fn=context_fn, preserve_rng_state=False
    )
    count += 1

    logging.info(f"Re-wrapped {count} modules with SAC mode '{mode}'")


def _compile_dit_blocks(video_processor):
    """Compile individual DiT blocks with torch.compile for fused kernels."""
    # Override TORCHDYNAMO_DISABLE if set
    os.environ.pop("TORCHDYNAMO_DISABLE", None)

    net = video_processor.model.net
    if hasattr(net, "base_model") and hasattr(net.base_model, "model"):
        blocks_owner = net.base_model.model
    else:
        blocks_owner = net

    for i, (block_id, block) in enumerate(blocks_owner.blocks.named_children()):
        compiled = torch.compile(block, mode="default")
        blocks_owner.blocks.register_module(block_id, compiled)

    logging.info(f"Compiled {len(blocks_owner.blocks)} DiT blocks with torch.compile")


def _build_param_groups(joint_model, video_lr, action_lr, repa_lr, weight_decay):
    """Build optimizer parameter groups with separate LRs and weight decay."""
    video_decay, video_no_decay = [], []
    action_decay, action_no_decay = [], []
    repa_decay, repa_no_decay = [], []

    for name, param in joint_model.video_processor.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in name or "norm" in name.lower():
            video_no_decay.append(param)
        else:
            video_decay.append(param)

    for name, param in joint_model.action_model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in name or "norm" in name.lower():
            action_no_decay.append(param)
        else:
            action_decay.append(param)

    if hasattr(joint_model, "repa_projectors") and joint_model.repa_projectors is not None:
        for name, param in joint_model.repa_projectors.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or "bias" in name:
                repa_no_decay.append(param)
            else:
                repa_decay.append(param)

    groups = [
        {"params": video_decay, "lr": video_lr, "weight_decay": weight_decay, "name": "video_dit"},
        {"params": video_no_decay, "lr": video_lr, "weight_decay": 0.0, "name": "video_dit_no_wd"},
        {"params": action_decay, "lr": action_lr, "weight_decay": weight_decay, "name": "action_head"},
        {"params": action_no_decay, "lr": action_lr, "weight_decay": 0.0, "name": "action_head_no_wd"},
        {"params": repa_decay, "lr": repa_lr, "weight_decay": weight_decay, "name": "repa_proj"},
        {"params": repa_no_decay, "lr": repa_lr, "weight_decay": 0.0, "name": "repa_proj_no_wd"},
    ]
    return [g for g in groups if len(g["params"]) > 0]


def main():
    args = parse_args()
    train_loop(args)


if __name__ == "__main__":
    main()
