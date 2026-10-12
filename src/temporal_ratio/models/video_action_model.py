"""Joint Video DiT + Action Head model with configurable gradient flow.

Training modes:
  1. JOINT_FULL: Full gradient flow from action loss through extracted features
     back into video DiT. Video loss + action loss, both update DiT.
  2. JOINT_DETACH: Video DiT trains via video loss, but features are detached
     before passing to action head. Action loss does NOT flow back to DiT.
  3. FROZEN_VIDEO: Video DiT is completely frozen. No video loss.
     Only the action head trains on extracted features.
  4. VIDEO_ONLY: only the video loss; the action head is frozen and not run.

Optional REPA (REPresentation Alignment) loss:
  Aligns intermediate DiT hidden states with frozen DINOv2/CLIP features
  of the last video frame (goal guidance). Piggybacks on the video forward
  pass — zero extra DiT compute.

Usage:
    model = JointVideoActionModel(
        action_model=..., video_processor=..., video_config=...,
        training_mode="joint_full",  # or "joint_detach" or "frozen_video"
        repa_config=REPAConfig(enabled=True, encoder_name="dinov2-vit-b"),
    )
"""

import logging
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

log = logging.getLogger(__name__)


class TrainingMode(Enum):
    JOINT_FULL = "joint_full"        # Full grads: action → features → video DiT
    JOINT_DETACH = "joint_detach"    # Video trains on video loss, features detached for action
    FROZEN_VIDEO = "frozen_video"    # Video frozen, only action head trains
    VIDEO_ONLY = "video_only"        # Only video loss, no action forward/loss at all


class JointVideoActionModel(nn.Module):
    """Joint Video DiT + VideoVLAPytorch action head.

    Implements DiT4DiT-style training with configurable gradient flow:
    1. Video DiT forward: sample tau_v, predict velocity, compute video loss
    2. Feature extraction: sample tau_f, extract hidden states from DiT
    3. Action head forward: compute action flow-matching loss
    4. Joint loss depends on training_mode
    """

    def __init__(
        self,
        action_model: nn.Module,
        video_processor: nn.Module,
        video_config,
        video_loss_weight: float = 1.0,
        training_mode: str = "joint_full",
        device: str | torch.device = "cuda",
        repa_config=None,
        state_dropout_rate: float = 0.0,
        first_frame_drop_prob: float = 0.0,
    ):
        """
        Args:
            action_model: VideoVLAPytorch instance.
            video_processor: VideoProcessor with .model (CosmosVideo2WorldModel).
            video_config: VideoDiTConfig with feature_layer, feature_dim, etc.
            video_loss_weight: Lambda for video loss (ignored in frozen_video mode).
            training_mode: One of "joint_full", "joint_detach", "frozen_video".
            device: Target device.
            repa_config: Optional REPAConfig for representation alignment loss.
            state_dropout_rate: Probability of zeroing out the state vector per sample during training.
            first_frame_drop_prob: Probability of hiding the first latent frame (the clean conditioning
                frame) from the action head per sample during training.
        """
        super().__init__()
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.training_mode = TrainingMode(training_mode)
        self.video_loss_weight = video_loss_weight
        self.state_dropout_rate = state_dropout_rate
        self.first_frame_drop_prob = first_frame_drop_prob
        self.feature_layer = video_config.feature_layer
        self.feature_dim = video_config.feature_dim
        self.num_feature_timestep_buckets = video_config.num_feature_timestep_buckets

        # Video processor
        self.video_processor = video_processor
        if hasattr(video_processor, "model"):
            self._cosmos_model = video_processor.model
        else:
            raise ValueError("video_processor must have a .model attribute (CosmosVideo2WorldModel)")

        # Configure gradient flow based on training mode
        self._setup_gradients()

        # Video latent shape
        nf = video_config.num_video_frames
        h = video_config.image_height
        w = video_config.image_width
        self.video_latent_shape = ((nf - 1) // 4 + 1, h // 16, w // 16)
        log.info(f"Video latent shape (T, H, W): {self.video_latent_shape}")

        # Action head — convert to bfloat16 for consistency with video DiT
        self.action_model = action_model.to(dtype=torch.bfloat16, device=self.device)
        self.action_model.set_video_latent_shape(self.video_latent_shape)

        # Freeze action head in VIDEO_ONLY mode (deferred from _setup_gradients)
        if self.training_mode == TrainingMode.VIDEO_ONLY:
            for param in self.action_model.parameters():
                param.requires_grad = False
            log.info("VIDEO_ONLY: action head frozen (%dM params)", sum(p.numel() for p in self.action_model.parameters()) // 1_000_000)

        # REPA (REPresentation Alignment) — optional goal-guidance loss
        self.repa_config = repa_config
        if repa_config is not None and repa_config.enabled:
            self._setup_repa(repa_config)
        else:
            self.repa_encoders = None
            self.repa_projectors = None

        self._log_params()

    def _setup_gradients(self):
        """Configure which parameters are trainable based on training mode.

        Matches the original Cosmos Predict2.5 training setup:
        - The entire DiT (self.net) is trainable (blocks, embedders, final layer)
        - Only the VAE (tokenizer) and conditioner (text encoder) are frozen
        """
        # Freeze everything first (VAE, conditioner, text encoder, etc.)
        for param in self.video_processor.parameters():
            param.requires_grad = False

        if self.training_mode == TrainingMode.FROZEN_VIDEO:
            log.info("FROZEN_VIDEO mode: video DiT frozen, only action head trains")
        elif self.training_mode == TrainingMode.VIDEO_ONLY:
            # Action head freeze deferred to _freeze_action_head() called after action_model is assigned
            log.info("VIDEO_ONLY mode: action head will be frozen, only video DiT trains")
        if self.training_mode in (TrainingMode.JOINT_FULL, TrainingMode.JOINT_DETACH,
                                    TrainingMode.VIDEO_ONLY):
            # Unfreeze DiT (net)
            model = self._cosmos_model
            if hasattr(model, "net"):
                net = model.net
                # Check if LoRA (PEFT) is active
                is_peft = hasattr(net, "base_model") and hasattr(net, "peft_config")
                if is_peft:
                    # Re-enable LoRA adapter params (blanket freeze above disabled them)
                    for name, param in net.named_parameters():
                        if "lora_" in name:
                            param.requires_grad = True
                    n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
                    n_total = sum(p.numel() for p in net.parameters())
                    log.info(
                        f"{self.training_mode.value} mode: DiT with LoRA "
                        f"({n_trainable/1e6:.1f}M trainable / {n_total/1e6:.0f}M total, "
                        f"VAE+conditioner frozen)"
                    )
                else:
                    # Full finetuning: unfreeze all DiT params
                    for param in net.parameters():
                        param.requires_grad = True
                    n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
                    log.info(
                        f"{self.training_mode.value} mode: entire DiT trainable "
                        f"({n_trainable/1e6:.0f}M params, VAE+conditioner frozen)"
                    )
            else:
                log.warning("Could not find video DiT net to unfreeze!")

    def _setup_repa(self, repa_config):
        """Initialize REPA encoders and projection heads.

        Supports multiple encoders (e.g. DINOv2 + CLIP simultaneously).
        Each encoder × layer combination gets its own projector MLP.
        """
        from temporal_ratio.models.repa_encoder import REPAEncoder

        # Load all requested encoders (frozen)
        self.repa_encoders = nn.ModuleList()
        self._repa_encoder_names = repa_config.encoder_names
        for enc_name in repa_config.encoder_names:
            self.repa_encoders.append(REPAEncoder(enc_name, device=self.device))

        # One projector per (encoder, layer) pair
        # Key format: "{encoder_idx}_{layer_idx}"
        self.repa_projectors = nn.ModuleDict()
        for enc_idx, encoder in enumerate(self.repa_encoders):
            for layer_idx in repa_config.repa_layers:
                key = f"{enc_idx}_{layer_idx}"
                self.repa_projectors[key] = nn.Sequential(
                    nn.Linear(self.feature_dim, repa_config.projector_dim),
                    nn.SiLU(),
                    nn.Linear(repa_config.projector_dim, repa_config.projector_dim),
                    nn.SiLU(),
                    nn.Linear(repa_config.projector_dim, encoder.output_dim),
                ).to(dtype=torch.float32, device=self.device)

        # Latent spatial dims for last-frame token slicing
        self._T_latent = self.video_latent_shape[0]
        self._H_latent = self.video_latent_shape[1]
        self._W_latent = self.video_latent_shape[2]

        enc_summary = ", ".join(f"{n}(dim={e.output_dim})" for n, e in
                                zip(repa_config.encoder_names, self.repa_encoders))
        log.info(
            f"REPA enabled: encoders=[{enc_summary}], "
            f"layers={repa_config.repa_layers}, proj_coeff={repa_config.proj_coeff}, "
            f"projectors={len(self.repa_projectors)}x MLP({self.feature_dim}->enc_dim)"
        )

    def compute_repa_loss(
        self,
        intermediate_features: list[torch.Tensor],
        last_frame_pixels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute REPA alignment loss on last-frame tokens (goal guidance).

        Args:
            intermediate_features: List of (B, T*H*W, D) tensors from DiT blocks.
            last_frame_pixels: (B, H, W, 3) uint8 or float — the last video frame.

        Returns:
            Scalar REPA loss (negative cosine similarity, averaged over encoders and layers).
        """
        HW = self._H_latent * self._W_latent
        last_t_start = (self._T_latent - 1) * HW
        last_t_end = self._T_latent * HW

        proj_loss = torch.tensor(0.0, device=self.device)
        num_terms = 0

        for enc_idx, encoder in enumerate(self.repa_encoders):
            # 1. Get encoder features for last frame (frozen, no grad)
            z_encoder = encoder(last_frame_pixels)  # (B, num_enc_patches, enc_dim)

            # 2. Interpolate encoder features to match DiT spatial grid
            # Infer spatial dims from num_patches (handles CLIP 224x224 resize)
            num_patches = z_encoder.shape[1]
            # For square CLIP: sqrt(256)=16. For DINOv2 on 224x448: 16*32=512
            enc_h = last_frame_pixels.shape[1] // encoder.patch_size
            enc_w = last_frame_pixels.shape[2] // encoder.patch_size
            # If encoder resized input (e.g. CLIP to 224x224), recalculate from actual patches
            if enc_h * enc_w != num_patches:
                import math
                enc_h = enc_w = int(math.sqrt(num_patches))
            z_enc_spatial = rearrange(z_encoder, "b (h w) d -> b d h w", h=enc_h, w=enc_w)
            z_enc_resized = F.interpolate(
                z_enc_spatial, size=(self._H_latent, self._W_latent),
                mode="bilinear", align_corners=False,
            )
            z_target = rearrange(z_enc_resized, "b d h w -> b (h w) d")
            z_target_norm = F.normalize(z_target.detach(), dim=-1)

            # 3. Compute loss for each DiT layer
            # intermediate_features can be a list (from DiT) or dict {layer_idx: tensor}
            for i, layer_idx in enumerate(self.repa_config.repa_layers):
                if isinstance(intermediate_features, dict):
                    feat = intermediate_features[layer_idx]
                else:
                    feat = intermediate_features[i]
                dit_last = feat[:, last_t_start:last_t_end, :]  # (B, H*W, D)

                key = f"{enc_idx}_{layer_idx}"
                z_proj = self.repa_projectors[key](dit_last.float())
                z_proj_norm = F.normalize(z_proj, dim=-1)

                proj_loss = proj_loss + (-(z_proj_norm * z_target_norm).sum(dim=-1).mean())
                num_terms += 1

        return proj_loss / max(num_terms, 1)

    def _log_params(self):
        video_total = sum(p.numel() for p in self.video_processor.parameters())
        video_train = sum(p.numel() for p in self.video_processor.parameters() if p.requires_grad)
        action_train = sum(p.numel() for p in self.action_model.parameters() if p.requires_grad)
        repa_train = 0
        if self.repa_projectors is not None:
            repa_train = sum(p.numel() for p in self.repa_projectors.parameters())
        log.info(
            f"Params: video={video_total/1e6:.0f}M (trainable={video_train/1e6:.0f}M), "
            f"action={action_train/1e6:.0f}M, repa_proj={repa_train/1e6:.1f}M, "
            f"mode={self.training_mode.value}"
        )

    @torch.no_grad()
    def tokenize_video(
        self,
        images: torch.Tensor,
        context_embedding: torch.Tensor | str | list[str] | None = None,
        num_conditional_frames: int = 1,
    ) -> tuple[torch.Tensor, object]:
        """Encode video frames to latents via frozen VAE."""
        return self.video_processor.tokenize_video(
            images, context_embedding, num_conditional_frames=num_conditional_frames
        )

    def video_forward(
        self,
        x0_latent: torch.Tensor,
        condition: object,
        last_frame_pixels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Video DiT forward: flow matching velocity prediction loss + optional REPA loss.

        Matches the original Cosmos Predict2.5 training pipeline exactly:
        1. Sample t ~ uniform(0, 1)
        2. Map to discrete timesteps via noise_scheduler
        3. Get sigmas from noise_scheduler (shift=3)
        4. Interpolate: x_t = epsilon * sigma + x0 * (1 - sigma)
        5. Predict velocity, compute MSE loss (uniform time weighting)

        Args:
            x0_latent: Clean video latent.
            condition: Condition dict.
            last_frame_pixels: (B, H, W, 3) last frame for REPA. None to skip REPA.

        Returns:
            (video_loss, repa_loss) — repa_loss is 0 if REPA disabled.
        """
        model = self._cosmos_model
        rf = model.rectified_flow
        tk_fp32 = model.tensor_kwargs_fp32
        tk = model.tensor_kwargs

        x0 = x0_latent.to(**tk_fp32)
        B = x0.shape[0]

        # 1. Sample noise
        epsilon = torch.randn(x0.size(), **tk_fp32)

        # 2. Sample training time t ~ U(0,1) and map to discrete timesteps + sigmas
        #    (exact same calls as text2world_model_rectified_flow.py forward())
        t_B = rf.sample_train_time(B).to(**tk_fp32)
        t_B = rearrange(t_B, "b -> b 1")
        timesteps = rf.get_discrete_timestamp(t_B, tk_fp32)
        sigmas = rf.get_sigmas(timesteps, tk_fp32)

        timesteps = rearrange(timesteps, "b -> b 1")
        sigmas = rearrange(sigmas, "b -> b 1")

        # 3. Interpolate: x_t = epsilon * sigma + x0 * (1 - sigma)
        xt, vt_target = rf.get_interpolation(epsilon, x0, sigmas)

        # 4. Request REPA intermediate features if enabled
        repa_layer_ids = None
        if self.repa_config and self.repa_config.enabled and last_frame_pixels is not None:
            repa_layer_ids = self.repa_config.repa_layers

        # 5. Denoise (same call as original: self.denoise(noise, xt, timesteps, condition))
        result = self.video_processor.compiled_denoiser(
            noise=epsilon,
            xt_B_C_T_H_W=xt.to(**tk),
            timesteps_B_T=timesteps,
            condition=condition,
            intermediate_feature_ids=repa_layer_ids,
        )

        if repa_layer_ids:
            vt_pred, intermediate_features = result
        else:
            vt_pred = result[0] if isinstance(result, (tuple, list)) else result
            intermediate_features = None

        # 6. Loss: MSE with uniform time weighting (matches original)
        per_instance_loss = torch.mean(
            (vt_pred.float() - vt_target.float()) ** 2,
            dim=list(range(1, vt_pred.dim())),
        )
        video_loss = torch.mean(per_instance_loss)

        # REPA loss (piggybacks on same DiT forward — zero extra compute)
        repa_loss = torch.tensor(0.0, device=self.device)
        if intermediate_features is not None and last_frame_pixels is not None:
            repa_loss = self.compute_repa_loss(intermediate_features, last_frame_pixels)

        return video_loss, repa_loss

    def video_forward_with_features(
        self,
        x0_latent: torch.Tensor,
        condition: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single DiT forward that computes BOTH video loss AND action features.

        Uses one shared noise/timestep sample for both objectives, saving a full
        DiT forward pass compared to calling video_forward + extract_features.

        Returns:
            (video_loss, features, t_B)
        """
        model = self._cosmos_model
        rf = model.rectified_flow
        tk_fp32 = model.tensor_kwargs_fp32
        tk = model.tensor_kwargs

        x0 = x0_latent.to(**tk_fp32)
        B = x0.shape[0]

        # Sample noise and timestep (shared for video loss + feature extraction)
        epsilon = torch.randn(x0.size(), **tk_fp32)
        t_B = rf.sample_train_time(B).to(**tk_fp32)
        t_B_unsq = rearrange(t_B, "b -> b 1")
        timesteps = rf.get_discrete_timestamp(t_B_unsq, tk_fp32)
        sigmas = rf.get_sigmas(timesteps, tk_fp32)

        timesteps = rearrange(timesteps, "b -> b 1")
        sigmas = rearrange(sigmas, "b -> b 1")

        xt, vt_target = rf.get_interpolation(epsilon, x0, sigmas)

        # Grad context for feature extraction depends on training mode
        if self.training_mode == TrainingMode.FROZEN_VIDEO:
            ctx = torch.no_grad()
        else:
            ctx = torch.enable_grad()

        with ctx:
            result = self.video_processor.compiled_denoiser(
                noise=epsilon,
                xt_B_C_T_H_W=xt.to(**tk),
                timesteps_B_T=timesteps,
                condition=condition,
                intermediate_feature_ids=[self.feature_layer],
            )
            vt_pred, intermediate_features = result

        # Video loss
        per_instance_loss = torch.mean(
            (vt_pred.float() - vt_target.float()) ** 2,
            dim=list(range(1, vt_pred.dim())),
        )
        video_loss = torch.mean(per_instance_loss)

        # Extract features
        if isinstance(intermediate_features, dict):
            features = intermediate_features[self.feature_layer]
        elif isinstance(intermediate_features, (list, tuple)):
            features = intermediate_features[0]
        else:
            features = intermediate_features

        if self.training_mode in (TrainingMode.JOINT_DETACH, TrainingMode.FROZEN_VIDEO):
            features = features.detach()

        return video_loss, features, t_B

    def extract_features(
        self,
        x0_latent: torch.Tensor,
        condition: object,
        tau_f: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract intermediate features from video DiT.

        Uses the model's rectified_flow scheduler for timestep/sigma mapping,
        matching the original Cosmos training pipeline.

        Gradient flow depends on training_mode:
          - JOINT_FULL: gradients flow through features back to DiT
          - JOINT_DETACH: features are detached (no grads to DiT from action loss)
          - FROZEN_VIDEO: no grads at all (torch.no_grad context)
        """
        model = self._cosmos_model
        rf = model.rectified_flow
        tk_fp32 = model.tensor_kwargs_fp32
        tk = model.tensor_kwargs

        x0 = x0_latent.to(**tk_fp32)
        B = x0.shape[0]

        # Sample t and map through scheduler (same as video_forward)
        if tau_f is None:
            t_B = rf.sample_train_time(B).to(**tk_fp32)
        else:
            t_B = tau_f.to(**tk_fp32)
        t_B_unsq = rearrange(t_B, "b -> b 1")
        timesteps_f = rf.get_discrete_timestamp(t_B_unsq, tk_fp32)
        sigmas_f = rf.get_sigmas(timesteps_f, tk_fp32)

        timesteps_f = rearrange(timesteps_f, "b -> b 1")
        sigmas_f = rearrange(sigmas_f, "b -> b 1")

        epsilon_f = torch.randn(x0.size(), **tk_fp32)
        xt_f, _ = rf.get_interpolation(epsilon_f, x0, sigmas_f)

        # Feature extraction — grad context depends on mode
        if self.training_mode == TrainingMode.FROZEN_VIDEO:
            ctx = torch.no_grad()
        else:
            ctx = torch.enable_grad()

        with ctx:
            _, intermediate_features = self.video_processor.compiled_denoiser(
                noise=epsilon_f,
                xt_B_C_T_H_W=xt_f.to(**tk),
                timesteps_B_T=timesteps_f,
                condition=condition,
                intermediate_feature_ids=[self.feature_layer],
            )

        # Features returned as list (one per requested layer) or dict
        if isinstance(intermediate_features, dict):
            features = intermediate_features[self.feature_layer]
        elif isinstance(intermediate_features, (list, tuple)):
            features = intermediate_features[0]  # first (and only) requested layer
        else:
            features = intermediate_features

        # Detach features if needed (JOINT_DETACH or FROZEN_VIDEO)
        if self.training_mode in (TrainingMode.JOINT_DETACH, TrainingMode.FROZEN_VIDEO):
            features = features.detach()

        return features, t_B

    def forward(
        self,
        observation,
        actions: torch.Tensor,
        action_latent: torch.Tensor,
        action_condition: object,
        video_latent: torch.Tensor | None = None,
        video_condition: object | None = None,
    ) -> dict[str, torch.Tensor]:
        """Joint forward pass with dual dataloader support.

        Args:
            observation: Observation dataclass (from action dataloader).
            actions: Ground truth actions (B, H, D) from action dataloader.
            action_latent: VAE-encoded video latent from action data (for feature extraction).
            action_condition: Condition from action data.
            video_latent: VAE-encoded video latent from video dataloader (for video loss).
                If None, uses action_latent for video loss too (single-dataloader mode).
            video_condition: Condition from video dataloader.
                If None, uses action_condition for video loss too.

        Returns dict with: total_loss, video_loss, action_loss
        """
        # Use action data as fallback if no separate video data provided
        if video_latent is None:
            video_latent = action_latent
        if video_condition is None:
            video_condition = action_condition

        # Extract last frame pixels for REPA goal guidance
        last_frame_pixels = None
        if self.repa_config and self.repa_config.enabled:
            if hasattr(observation, 'wan_images_latents') and observation.wan_images_latents is not None:
                last_frame_pixels = observation.wan_images_latents[:, -1]  # (B, H, W, C)

        # Video loss + REPA loss (skip in frozen_video mode)
        if self.training_mode not in (TrainingMode.FROZEN_VIDEO,):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                video_loss, repa_loss = self.video_forward(
                    video_latent, video_condition, last_frame_pixels=last_frame_pixels,
                )
        else:
            video_loss = torch.tensor(0.0, device=self.device)
            repa_loss = torch.tensor(0.0, device=self.device)

        # Action loss (skip in video_only mode)
        if self.training_mode != TrainingMode.VIDEO_ONLY:
            # Feature extraction from ACTION data + action loss
            features, tau_f = self.extract_features(action_latent, action_condition)
            self._last_features = features.detach()
            self._last_tau_f = tau_f.detach()

            # Cast observation state and actions to float32 (autocast will handle bf16)
            import dataclasses as dc
            if hasattr(observation, 'state') and observation.state.dtype == torch.float64:
                observation = dc.replace(observation, state=observation.state.float())
            actions = actions.float()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                action_losses = self.action_model(observation, actions, features, tau_f,
                                                   state_dropout_rate=self.state_dropout_rate,
                                                   first_frame_drop_prob=self.first_frame_drop_prob)
            action_loss = action_losses.mean()
        else:
            action_loss = torch.tensor(0.0, device=self.device)

        # Total loss
        proj_coeff = self.repa_config.proj_coeff if (self.repa_config and self.repa_config.enabled) else 0.0
        if self.training_mode == TrainingMode.FROZEN_VIDEO:
            total_loss = action_loss
        elif self.training_mode == TrainingMode.VIDEO_ONLY:
            total_loss = self.video_loss_weight * video_loss + proj_coeff * repa_loss
        else:
            total_loss = action_loss + self.video_loss_weight * video_loss + proj_coeff * repa_loss

        return {
            "total_loss": total_loss,
            "video_loss": video_loss.detach(),
            "action_loss": action_loss.detach(),
            "repa_loss": repa_loss.detach(),
        }

    @torch.no_grad()
    def sample_actions(
        self,
        device,
        observation,
        images_latent: torch.Tensor,
        condition: object,
        tau_f: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Sample actions via Euler ODE integration (inference)."""
        import dataclasses as dc
        if hasattr(observation, 'state') and observation.state.dtype == torch.float64:
            observation = dc.replace(observation, state=observation.state.float())
        B = images_latent.shape[0]
        if tau_f is None:
            tau_f = torch.full((B,), 0.5, device=self.device)
        features, tau_f = self.extract_features(images_latent, condition, tau_f=tau_f)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return self.action_model.sample_actions(
                device=device, observation=observation,
                video_latents=features, video_timestep=tau_f,
                noise=noise, num_steps=num_steps,
            )

    def get_parameter_groups(self, video_lr: float, action_lr: float, repa_lr: float | None = None) -> list[dict]:
        """Return optimizer param groups with separate LRs."""
        groups = []
        video_params = [p for p in self.video_processor.parameters() if p.requires_grad]
        action_params = [p for p in self.action_model.parameters() if p.requires_grad]
        if video_params:
            groups.append({"params": video_params, "lr": video_lr, "name": "video_dit"})
        if action_params:
            groups.append({"params": action_params, "lr": action_lr, "name": "action_head"})
        if self.repa_projectors is not None:
            repa_params = list(self.repa_projectors.parameters())
            if repa_params:
                groups.append({"params": repa_params, "lr": repa_lr or action_lr, "name": "repa_proj"})
        return groups
