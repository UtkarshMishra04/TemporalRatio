"""Command-line interface of ``temporal_ratio.training.train`` (argparse only, importable without torch).

Kept separate so configuration tooling and tests can build and validate trainer invocations
without importing the training stack.
"""

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Joint Cosmos Video DiT + Action Head training")
    p.add_argument("data_config_name", type=str)
    p.add_argument("--exp_name", type=str, default="joint_cosmos_action")

    # Training mode
    p.add_argument("--training_mode", type=str, default="joint_full",
                   choices=["joint_full", "joint_detach", "frozen_video", "video_only"],
                   help="joint_full: full grads action→video; joint_detach: video trains separately; frozen_video: only action trains; video_only: only video trains, no action loss")

    # Loss
    p.add_argument("--video_loss_weight", type=float, default=1.0, help="lambda in L_action + lambda*L_video")

    # Learning rates (DiT4DiT Table 4)
    p.add_argument("--video_lr", type=float, default=1e-5, help="VGM LR (DiT4DiT: 1e-5)")
    p.add_argument("--action_lr", type=float, default=1e-4, help="Action Model LR (DiT4DiT: 1e-4)")

    # Schedule (DiT4DiT Table 4)
    p.add_argument("--max_train_steps", type=int, default=100_000)
    p.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup steps = ratio * max_steps (DiT4DiT: 5000/100000=0.05)")
    p.add_argument("--min_lr_ratio", type=float, default=0.005, help="Min LR = ratio * peak (DiT4DiT: 5e-7/1e-4=0.005)")

    # Batch
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1,
                   help="Accumulate gradients over N micro-batches")

    # Checkpointing
    p.add_argument("--save_interval", type=int, default=2000)
    p.add_argument("--keep_every", type=int, default=0,
                   help="Also save at every multiple of this many steps (e.g. one epoch) and never delete "
                        "those checkpoints; others are pruned to the most recent ones. 0 = off.")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/cosmos_action_joint")

    # Feature extraction
    p.add_argument("--feature_layer", type=int, default=18)
    p.add_argument("--num_feature_timestep_buckets", type=int, default=1000,
                   help="DiT4DiT Table 4: Num Timestep Buckets = 1000")

    # EMA
    p.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay for constant EMA (0=disabled)")
    p.add_argument("--ema_type", type=str, default="power", choices=["constant", "power"],
                   help="EMA type: constant (fixed decay) or power (Cosmos iteration-dependent)")
    p.add_argument("--ema_rate", type=float, default=0.10, help="Power EMA rate parameter (Cosmos default: 0.10)")

    # Optimizer (Cosmos defaults)
    p.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay (Cosmos LoRA: 0.001, Cosmos base: 0.1)")
    p.add_argument("--gradient_clip_norm", type=float, default=1.0)
    p.add_argument("--beta2", type=float, default=0.99, help="AdamW beta2 (Cosmos: 0.99)")

    # Wandb
    p.add_argument("--wandb_enabled", action="store_true", default=True)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--project_name", type=str, default="cosmos-action-joint")

    # Resume/overwrite
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--action_head_weight_path", type=str, default=None)
    p.add_argument("--causal", action="store_true")

    # Action model
    p.add_argument("--action_dim", type=int, default=32)
    p.add_argument("--state_dim", type=int, default=None, help="State input dim (defaults to action_dim)")
    p.add_argument("--state_dropout_rate", type=float, default=0.0,
                   help="Per-sample probability of zeroing state input during training. "
                        "Forces the action head to rely more on visual features.")
    p.add_argument("--action_first_frame_drop_prob", type=float, default=0.0,
                   help="Per-sample probability of hiding the first latent frame (the clean conditioning "
                        "frame) from the action head during training, so it relies on the predicted frames.")
    p.add_argument("--action_horizon", type=int, default=17)
    p.add_argument("--observation_horizon", type=int, default=17)

    # Base Cosmos checkpoint (DiT4DiT: always start from base pretrained, not dataset-specific)
    p.add_argument("--base_cosmos_ckpt", type=str, default=None,
                   help="Path to base Cosmos pretrained checkpoint. If None, auto-downloads from HuggingFace.")

    # Model size

    # Video dataset for video DiT loss (separate from action data)
    p.add_argument("--video_dataset", type=str, default=None,
                   help="Video dataset for DiT loss. None = use action data for video loss too.")

    # Data
    p.add_argument("--repo_id", type=str, default="physical-intelligence/libero")
    p.add_argument("--dataset_type", type=str, default="libero")

    # Video generation callback
    p.add_argument("--video_log_interval", type=int, default=1000)
    p.add_argument("--video_gen_steps", type=int, default=35)
    p.add_argument("--video_autoregressive_steps", type=int, default=10)

    # ── Speed optimizations ──
    p.add_argument("--no_gradient_checkpointing", action="store_true",
                   help="Disable DiT selective activation checkpointing (trades memory for speed)")

    # LoRA (Low-Rank Adaptation) for video DiT
    p.add_argument("--lora_enabled", action="store_true",
                   help="Enable LoRA adapters on the video DiT (parameter-efficient fine-tuning)")
    p.add_argument("--lora_rank", type=int, default=32, help="LoRA rank (default: 32)")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling (default: 32)")
    p.add_argument("--lora_target_modules", type=str,
                   default="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
                   help="Comma-separated list of DiT modules to apply LoRA to")
    p.add_argument("--lora_lr", type=float, default=None,
                   help="LR for LoRA params (default: same as video_lr)")

    # FSDP (Fully Sharded Data Parallel) for large models (14B+)
    p.add_argument("--fsdp", action="store_true",
                   help="Enable FSDP for the video DiT (required for 14B+ models)")
    p.add_argument("--fsdp_sharding_strategy", type=str, default="FULL_SHARD",
                   choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"],
                   help="FSDP sharding strategy (default: FULL_SHARD)")

    # Text dropout is handled by the conditioner's built-in 0.2 rate (from Cosmos config).
    # The --text_dropout_rate arg is DEPRECATED: it was stacking with conditioner dropout
    # giving effective ~36% instead of intended 20%. Kept for backward compat but ignored.
    p.add_argument("--text_dropout_rate", type=float, default=0.0,
                   help="DEPRECATED: conditioner handles text dropout (0.2). This arg is ignored.")

    # LR schedule
    p.add_argument("--lr_schedule", type=str, default="cosine", choices=["cosine", "cosmos"],
                   help="LR schedule: cosine (original openpi) or cosmos (Cosmos LambdaLinear)")
    p.add_argument("--lr_f_max", type=float, default=0.5, help="Cosmos schedule: peak LR factor (default: 0.5)")
    p.add_argument("--lr_f_min", type=float, default=0.2, help="Cosmos schedule: min LR factor (default: 0.2)")

    # fp32 LoRA params (matches Cosmos FusedAdamW master weights behavior)
    p.add_argument("--fp32_lora_params", action="store_true", default=True,
                   help="Upcast LoRA params to fp32 for optimizer (matches Cosmos)")
    p.add_argument("--no_fp32_lora_params", dest="fp32_lora_params", action="store_false")

    # Mixed conditional frames (Cosmos trains text2world/image2world/video2world jointly)
    p.add_argument("--mixed_conditional_frames", action="store_true", default=False,
                   help="Sample num_conditional_frames from {0,1,2} during training (Cosmos default)")
    p.add_argument("--conditional_frames_probs", type=str, default="0.333,0.333,0.334",
                   help="Probabilities for 0,1,2 conditional frames (Cosmos: equal)")

    # REPA (REPresentation Alignment) — goal-guidance loss
    p.add_argument("--repa_enabled", action="store_true",
                   help="Enable REPA alignment loss (aligns DiT features with DINO/CLIP of last frame)")
    p.add_argument("--repa_encoder", type=str, nargs="+", default=["dinov2-vit-b"],
                   help="Pretrained encoder(s) for REPA targets (e.g. dinov2-vit-b clip-vit-l)")
    p.add_argument("--repa_layers", type=int, nargs="+", default=[8],
                   help="DiT block indices for REPA alignment (e.g. 4 8 12)")
    p.add_argument("--repa_coeff", type=float, default=0.5, help="REPA loss weight (proj_coeff)")
    p.add_argument("--repa_lr", type=float, default=1e-4, help="LR for REPA projection heads")
    p.add_argument("--repa_projector_dim", type=int, default=2048, help="Hidden dim in REPA projection MLP")

    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)
