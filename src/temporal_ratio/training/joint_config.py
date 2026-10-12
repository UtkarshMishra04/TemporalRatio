"""Configuration dataclasses for joint Video DiT + Action Head training.

Defaults match DiT4DiT paper Table 4 exactly.
"""

from dataclasses import dataclass, field


@dataclass
class ActionDiTConfig:
    """Config for the ActionDiT (separate DiT for actions)."""
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    action_dim: int = 32
    state_dim: int = 32
    action_horizon: int = 17
    cross_attention_dim: int = 2048


@dataclass
class VideoDiTConfig:
    """Video DiT config — matches DiT4DiT Table 4 (Video DiT section)."""
    setup: str = "libero_flat_short"
    ckpt_path: str | None = None

    # DiT4DiT Table 4: Extract Layer = 18, Hidden Feature Dim = 2048
    feature_layer: int = 18
    feature_dim: int = 2048

    # DiT4DiT: tau_f sampled from discrete uniform U{0/T, ..., T/T}
    # Paper uses Num Timestep Buckets = 1000
    num_feature_timestep_buckets: int = 1000

    causal_temporal: bool = False
    image_height: int = 224
    image_width: int = 448
    num_video_frames: int = 17


@dataclass
class REPAConfig:
    """REPA (REPresentation Alignment) loss configuration.

    Aligns DiT intermediate hidden states with frozen pretrained encoder
    features of the last video frame (goal guidance).

    Supports multiple encoders: encoder_names is a list (e.g. ["dinov2-vit-b", "clip-vit-l"]).
    Each encoder gets its own set of projection heads. The REPA loss is averaged over all encoders.
    """
    enabled: bool = False
    encoder_names: list[str] = field(default_factory=lambda: ["dinov2-vit-b"])  # dinov2-vit-b/l/g, clip-vit-l
    repa_layers: list[int] = field(default_factory=lambda: [8])  # DiT block indices
    proj_coeff: float = 0.5             # REPA loss weight
    projector_dim: int = 2048           # Hidden dim in projection MLP
    repa_lr: float = 1e-4              # LR for projection heads


@dataclass
class JointTrainConfig:
    """Training config — matches DiT4DiT paper Table 4 exactly.

    DiT4DiT Table 4 Training Configurations:
      Per Device Batch Size: 8
      Num of GPUs: 32
      Max Train Steps: 100000
      Num Warmup Steps: 5000
      Warmup Ratio: 0.1
      VGM LR: 1e-5
      Action Model LR: 1e-4
      LR Scheduler: cosine with min lr
      Min LR: 5e-7
      Gradient Clipping: 1.0
      Gradient Accumulation Steps: 1
      Optimizer: AdamW
      beta1, beta2: (0.9, 0.95)
      eps: 1e-8
      Weight Decay: 1e-8
    """
    video_dit: VideoDiTConfig = field(default_factory=VideoDiTConfig)

    # --- Loss ---
    # DiT4DiT Eq. 10: L_total = L_action + lambda * L_video
    video_loss_weight: float = 1.0

    # --- Learning rates (DiT4DiT Table 4) ---
    video_lr: float = 1e-5    # VGM LR
    action_lr: float = 1e-4   # Action Model LR

    # --- Schedule (DiT4DiT Table 4) ---
    warmup_steps: int = 5000          # Num Warmup Steps
    max_train_steps: int = 100_000    # Max Train Steps
    min_lr: float = 5e-7             # Min LR

    # --- Optimizer (DiT4DiT Table 4) ---
    optimizer_betas: tuple[float, float] = (0.9, 0.95)  # beta1, beta2
    optimizer_eps: float = 1e-8                          # eps
    weight_decay: float = 1e-8                           # Weight Decay
    gradient_clip_norm: float = 1.0                      # Gradient Clipping

    # --- Batch (DiT4DiT Table 4) ---
    batch_size: int = 8               # Per Device Batch Size

    # --- Data ---
    data_config_name: str = "videoVLA_libero_short"
    # openpi config name — used by create_trained_policy for loading
    name: str = "videoVLA_libero_short"

    # --- Logging ---
    exp_name: str = "joint_cosmos_action"
    project_name: str = "cosmos-action-joint"
    checkpoint_dir: str = "checkpoints/cosmos_action_joint"
    log_interval: int = 10
    save_interval: int = 2000
    wandb_enabled: bool = True

    precision: str = "bfloat16"
    seed: int = 42
    resume: bool = False
    overwrite: bool = False
    action_head_weight_path: str | None = None

    @property
    def video_latent_shape(self) -> tuple[int, int, int]:
        nf = self.video_dit.num_video_frames
        h = self.video_dit.image_height
        w = self.video_dit.image_width
        return ((nf - 1) // 4 + 1, h // 16, w // 16)
