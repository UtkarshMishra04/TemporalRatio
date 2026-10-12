"""Command-line interface of ``temporal_ratio.training.train_dual_resolution`` (argparse only, importable without torch).

Kept separate so configuration tooling and tests can build and validate trainer invocations
without importing the training stack.
"""

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dual-resolution joint training: long-horizon video + short-horizon action"
    )
    p.add_argument("data_config_name", type=str,
                   help="openpi config name for 17-frame action dataloader (e.g. videoVLA_libero_short)")
    p.add_argument("--exp_name", type=str, default="joint_93f")

    # ── Dual resolution ──
    p.add_argument("--video_num_frames", type=int, default=93,
                   help="Number of frames for video loss (default: 93 → 24 temporal latent tokens)")
    p.add_argument("--video_frame_skip", type=int, default=1,
                   help="Frame skip for video dataloader (default: 1)")
    p.add_argument("--video_batch_size", type=int, default=2,
                   help="Per-GPU batch size for long-horizon video data (keep small for memory)")
    p.add_argument("--action_batch_size", type=int, default=4,
                   help="Per-GPU batch size for 17-frame action data")
    p.add_argument("--video_dataset", type=str, default="libero",
                   help="Video dataset for long-horizon DiT loss")
    p.add_argument("--val_split", type=float, default=0.0,
                   help="Fraction of episodes held out for val nL2 (0 to disable, default 0)")
    p.add_argument("--val_seed", type=int, default=42,
                   help="Seed for val split selection")

    # ── Training mode ──
    p.add_argument("--training_mode", type=str, default="joint_full",
                   choices=["joint_full", "joint_detach", "frozen_video", "video_only"])
    p.add_argument("--video_loss_weight", type=float, default=1.0)
    p.add_argument("--video_aux_every_k", type=int, default=1,
                   help="Run 93f auxiliary video loss every K steps (1=every step, 4=every 4th step). "
                        "Action loss (17f) still runs every step.")

    # ── Learning rates ──
    p.add_argument("--video_lr", type=float, default=1e-5)
    p.add_argument("--action_lr", type=float, default=1e-4)

    # ── Schedule ──
    p.add_argument("--max_train_steps", type=int, default=100_000)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--min_lr_ratio", type=float, default=0.005)

    # ── Batch / accumulation ──
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)

    # ── Checkpointing ──
    p.add_argument("--save_interval", type=int, default=2000)
    p.add_argument("--keep_every", type=int, default=0,
                   help="Also save at every multiple of this many steps (e.g. one epoch) and never delete "
                        "those checkpoints; others are pruned to the most recent ones. 0 = off.")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/cosmos_action_joint_93f")

    # ── Feature extraction ──
    p.add_argument("--feature_layer", type=int, default=18)
    p.add_argument("--num_feature_timestep_buckets", type=int, default=1000)

    # ── EMA ──
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--ema_type", type=str, default="power", choices=["constant", "power"],
                   help="EMA type: 'constant' (fixed decay) or 'power' (Cosmos iteration-dependent)")
    p.add_argument("--ema_rate", type=float, default=0.10,
                   help="PowerEMA rate parameter (only used when --ema_type=power)")

    # ── Optimizer ──
    p.add_argument("--weight_decay", type=float, default=0.001)
    p.add_argument("--beta2", type=float, default=0.99,
                   help="AdamW beta2 (Cosmos default: 0.99)")
    p.add_argument("--gradient_clip_norm", type=float, default=1.0)

    # ── Wandb ──
    p.add_argument("--wandb_enabled", action="store_true", default=True)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--project_name", type=str, default="cosmos-action-joint")

    # ── Resume / overwrite ──
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--action_head_weight_path", type=str, default=None)
    p.add_argument("--causal", action="store_true",
                    help="Causal temporal attention for DiT")

    # ── Action model ──
    p.add_argument("--action_dim", type=int, default=32)
    p.add_argument("--state_dim", type=int, default=None)
    p.add_argument("--state_dropout_rate", type=float, default=0.0,
                   help="Per-sample probability of zeroing state input during training.")
    p.add_argument("--action_first_frame_drop_prob", type=float, default=0.0,
                   help="Per-sample probability of hiding the first latent frame from the action head.")
    p.add_argument("--action_horizon", type=int, default=17)
    p.add_argument("--observation_horizon", type=int, default=17)

    # ── Base checkpoint ──
    p.add_argument("--base_cosmos_ckpt", type=str, default=None)

    # ── LoRA ──
    p.add_argument("--lora_enabled", action="store_true")
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_target_modules", type=str,
                   default="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2")
    p.add_argument("--lora_lr", type=float, default=None)

    # ── Condition dropout (DEPRECATED — conditioner handles it) ──
    p.add_argument("--text_dropout_rate", type=float, default=0.0,
                   help="DEPRECATED: conditioner already applies dropout_rate=0.2. This arg is ignored.")

    # ── LR schedule ──
    p.add_argument("--lr_schedule", type=str, default="cosine", choices=["cosine", "cosmos"],
                   help="LR schedule: 'cosine' (original) or 'cosmos' (Cosmos LambdaLinear)")
    p.add_argument("--lr_f_max", type=float, default=0.5,
                   help="Cosmos LR peak fraction (default 0.5 = 50%% of base LR)")
    p.add_argument("--lr_f_min", type=float, default=0.2,
                   help="Cosmos LR min fraction (default 0.2 = 20%% of base LR)")

    # ── fp32 LoRA params ──
    p.add_argument("--fp32_lora_params", action="store_true", default=True,
                   help="Upcast LoRA parameters to fp32 (matches Cosmos FusedAdamW master_weights)")
    p.add_argument("--no_fp32_lora_params", dest="fp32_lora_params", action="store_false")

    # ── Mixed conditional frames ──
    p.add_argument("--mixed_conditional_frames", action="store_true", default=False,
                   help="Train with mixed conditional frames (0=text2world, 1=image2world, 2=video2world)")
    p.add_argument("--conditional_frames_probs", type=str, default="0.333,0.333,0.334",
                   help="Comma-separated probs for 0,1,2 conditional frames")

    # ── FSDP ──
    p.add_argument("--fsdp", action="store_true")
    p.add_argument("--fsdp_sharding_strategy", type=str, default="FULL_SHARD",
                   choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"])

    # ── REPA ──
    p.add_argument("--repa_enabled", action="store_true")
    p.add_argument("--repa_encoder", type=str, nargs="+", default=["dinov2-vit-b"])
    p.add_argument("--repa_layers", type=int, nargs="+", default=[8])
    p.add_argument("--repa_coeff", type=float, default=0.5)
    p.add_argument("--repa_lr", type=float, default=1e-4)
    p.add_argument("--repa_projector_dim", type=int, default=2048)

    # ── Video generation callback ──
    p.add_argument("--video_log_interval", type=int, default=2000)
    p.add_argument("--video_gen_steps", type=int, default=35)
    p.add_argument("--video_autoregressive_steps", type=int, default=10)

    # ── Speed optimizations ──
    p.add_argument("--no_gradient_checkpointing", action="store_true",
                   help="Disable DiT selective activation checkpointing (trades memory for speed)")
    p.add_argument("--sac_mode", type=str, default=None,
                   choices=["none", "predict2_2b_720", "predict2_2b_720_aggressive", "block_wise", "mm_only"],
                   help="Override SAC mode (default: keep model's built-in aggressive mode)")
    p.add_argument("--compile", action="store_true",
                   help="Enable torch.compile on DiT blocks")
    p.add_argument("--video_num_workers", type=int, default=8,
                   help="Number of dataloader workers for video data")
    p.add_argument("--video_prefetch_factor", type=int, default=None,
                   help="Prefetch factor for video dataloader (default: PyTorch default=2)")
    p.add_argument("--persistent_workers", action="store_true",
                   help="Keep video dataloader workers alive between epochs")

    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)
