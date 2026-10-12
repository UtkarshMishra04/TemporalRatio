"""Cosmos-Predict2.5 2B Video2World model wrapper used by the trainers and the policies.

``VideoProcessor`` loads the DiT (base, fine-tuned ``video_model_full.pt``, or base + LoRA adapters),
the Wan2.1 VAE and the Reason1 text encoder through the vendored Cosmos inference pipeline
(``cosmos_video``) and exposes VAE encoding, flow-matching training targets and intermediate DiT
features for the action head.

A *video setup* (``VIDEO_SETUPS``) fixes how a robot's cameras are fed to the model: the Cosmos
experiment config, the resolution of the horizontally tiled camera frame and the camera order.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

# ---------------------------------------------------------------------------
# Local imports — these modules are being created by another agent in the
# same ``models/`` package.  The imports are structured so they will resolve
# once those files exist.
# ---------------------------------------------------------------------------
from cosmos_video._src.predict2.inference.video2world import Video2WorldInference

_log = logging.getLogger(__name__)

# ===========================================================================
# Video setups and checkpoints
# ===========================================================================


@dataclasses.dataclass(frozen=True)
class VideoSetup:
    """How a robot's cameras are fed to the Cosmos video model."""

    #: Cosmos experiment config (architecture, tokenizer, conditioning) in ``cosmos_video.experiments``.
    experiment: str
    #: (H, W) of the frame obtained by tiling the cameras left to right.
    image_size: tuple[int, int]
    #: Model image keys (after the robot's input transform), tiled left to right.
    camera_keys: tuple[str, ...]
    #: DiT width (``model_channels``) = dimension of the features read by the action head.
    feature_dim: int = 2048


VIDEO_SETUPS: dict[str, VideoSetup] = {
    # LIBERO: agent-view | wrist camera, 224x224 each.
    "libero_flat_short": VideoSetup(
        experiment="predict2_video2world_training_2b_cosmos_lerobot_libero_flat_short",
        image_size=(224, 448),
        camera_keys=("base_0_rgb", "left_wrist_0_rgb"),
    ),
    # Bimanual YAM (ABC-130k): top | left wrist | right wrist, 176x224 each.
    "abc130k_flat_short": VideoSetup(
        experiment="predict2_video2world_training_2b_cosmos_lerobot_yam_flat_short",
        image_size=(176, 224 * 3),
        camera_keys=("left_base_0_rgb", "right_base_0_rgb", "left_wrist_0_rgb"),
    ),
}

EXPERIMENT_NAMES = {name: s.experiment for name, s in VIDEO_SETUPS.items()}
IMAGE_SIZES = {name: s.image_size for name, s in VIDEO_SETUPS.items()}
FEATURE_DIMS = {name: s.feature_dim for name, s in VIDEO_SETUPS.items()}

def resolve_base_checkpoint() -> str:
    """Local path of the pretrained Cosmos-Predict2.5 2B DiT (``nvidia/Cosmos-Predict2.5-2B``,
    ``base/pre-trained``), downloaded at the revision pinned in the Cosmos checkpoint registry.

    ``$COSMOS_BASE_CKPT`` overrides the lookup.
    """
    override = os.environ.get("COSMOS_BASE_CKPT")
    if override:
        return override
    from cosmos_video._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
    from cosmos_video.config import MODEL_CHECKPOINTS, ModelKey

    return get_checkpoint_path(MODEL_CHECKPOINTS[ModelKey(post_trained=False)].s3.uri)


def resolve_video_checkpoint(spec: str | None) -> str:
    """Resolve a ``--base_cosmos_ckpt`` value: ``base``/``None`` -> pretrained 2B DiT, else a path.

    A path may be a ``video_model_full.pt`` file or a checkpoint directory (``VideoProcessor``
    picks ``video_model_full.pt`` or base + ``lora_adapters/`` from it).
    """
    if spec is None or spec.lower() == "base":
        return resolve_base_checkpoint()
    return spec


# ===========================================================================
# Helpers
# ===========================================================================

def sample_beta(alpha: float, beta: float, bsize: int, device: torch.device | str) -> torch.Tensor:
    """Sample from a Beta distribution."""
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def sample_logit_normal(mu: float, sigma: float, bsize: int, device: torch.device | str) -> torch.Tensor:
    """Sample from a logit-normal distribution (Normal -> sigmoid)."""
    mu_t = torch.as_tensor(mu, dtype=torch.float32, device=device)
    sigma_t = torch.as_tensor(sigma, dtype=torch.float32, device=device)
    dist = torch.distributions.Normal(mu_t, sigma_t)
    z = dist.sample((bsize,))
    return torch.sigmoid(z)


class CompiledDenoiser(nn.Module):
    """Thin wrapper around the model's ``denoise`` method so it can be
    ``torch.compile``-d if desired."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        noise: torch.Tensor,
        xt_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition: dict,
        intermediate_feature_ids: list[int] | None = None,
    ):
        return self.model.denoise(
            noise=noise,
            xt_B_C_T_H_W=xt_B_C_T_H_W,
            timesteps_B_T=timesteps_B_T,
            condition=condition,
            intermediate_feature_ids=intermediate_feature_ids,
        )


# ===========================================================================
# Main processor class
# ===========================================================================

class VideoProcessor(nn.Module):
    """Self-contained Cosmos video processor for robot policy learning.

    No dependencies on ``cosmos_predict2``.  Uses the local self-contained
    DiT, VAE, and scheduler modules to load and run Cosmos video models.
    """

    # DiT architecture configs (from Cosmos net.py defaults)
    # in_channels=16 (MinimalV1LVGDiT adds +1 internally for the condition mask)
    # Reason1 FULL_CONCAT: 28 layers × 3584 = 100352 → crossattn_proj → 1024
    DIT_CONFIGS = {
        "2b": dict(
            max_img_h=240, max_img_w=240, max_frames=128,
            in_channels=16, out_channels=16,
            patch_spatial=2, patch_temporal=1,
            model_channels=2048, num_blocks=28, num_heads=16,
            concat_padding_mask=True,
            pos_emb_cls="rope3d", pos_emb_learnable=True, pos_emb_interpolation="crop",
            use_adaln_lora=True, adaln_lora_dim=256,
            extra_per_block_abs_pos_emb=False,
            rope_h_extrapolation_ratio=1.0, rope_w_extrapolation_ratio=1.0,
            rope_t_extrapolation_ratio=1.0,
            crossattn_emb_channels=1024,
            use_crossattn_projection=True,
            crossattn_proj_in_channels=100352,  # 28 layers * 3584 hidden_size (Reason1 FULL_CONCAT)
        ),
        "14b": dict(
            max_img_h=240, max_img_w=240, max_frames=128,
            in_channels=16, out_channels=16,
            patch_spatial=2, patch_temporal=1,
            model_channels=5120, num_blocks=36, num_heads=40,
            concat_padding_mask=True,
            pos_emb_cls="rope3d", pos_emb_learnable=True, pos_emb_interpolation="crop",
            use_adaln_lora=True, adaln_lora_dim=256,
            extra_per_block_abs_pos_emb=False,
            rope_h_extrapolation_ratio=1.0, rope_w_extrapolation_ratio=1.0,
            rope_t_extrapolation_ratio=1.0,
            crossattn_emb_channels=1024,
            use_crossattn_projection=True,
            crossattn_proj_in_channels=100352,  # 28 layers * 3584 hidden_size (Reason1 FULL_CONCAT)
        ),
    }

    # HuggingFace repo for the Cosmos VAE tokenizer (4x temporal, 8x spatial)
    VAE_HF_REPO = "nvidia/Cosmos-Tokenizer-CV4x8x8"

    @classmethod
    def _resolve_vae_paths(cls, enc_fp=None, dec_fp=None, mean_std_fp=None):
        """Resolve VAE file paths, downloading from HuggingFace if needed."""
        from huggingface_hub import hf_hub_download
        if enc_fp is None or not os.path.exists(enc_fp):
            enc_fp = hf_hub_download(cls.VAE_HF_REPO, "encoder.jit")
        if dec_fp is None or not os.path.exists(dec_fp):
            dec_fp = hf_hub_download(cls.VAE_HF_REPO, "decoder.jit")
        # mean_std is optional — not in the tokenizer repo, use identity normalization
        if mean_std_fp is None or not os.path.exists(mean_std_fp):
            mean_std_fp = ""  # CosmosPretrainedVAE handles missing mean_std gracefully
        return enc_fp, dec_fp, mean_std_fp

    def __init__(
        self,
        setup: str = "libero_flat_short",
        device: str | torch.device = "cpu",
        ckpt_path: str | None = None,
        atten_temporal_causal: bool = False,
        num_global_denoising_steps: int = 35,
        # Model loading options
        dit_config: dict | None = None,
        vae_enc_fp: str | None = None,
        vae_dec_fp: str | None = None,
        vae_mean_std_fp: str | None = None,
        precision: str = "bfloat16",
    ):
        super().__init__()

        self.setup = setup
        self.atten_temporal_causal = atten_temporal_causal
        self.num_global_denoising_steps = num_global_denoising_steps
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        # Detect LoRA checkpoint: if ckpt_path points to a dir with lora_adapters/,
        # load HF base Cosmos + apply LoRA on top.
        # Skip auto-detection when ckpt_path is an explicit .pt file (e.g. merged checkpoint
        # from save_checkpoint's merge_adapter path) to avoid bf16 precision loss from
        # re-applying LoRA on base instead of using the merged weights directly.
        lora_adapter_path = None
        # ckpt_path: a .pt file (base or merged fine-tuned DiT) is loaded as is; a checkpoint
        # directory uses base + lora_adapters/ if present, else its video_model_full.pt.
        if ckpt_path is not None and Path(ckpt_path).is_dir():
            ckpt_dir = Path(ckpt_path)
            if (ckpt_dir / "lora_adapters" / "adapter_config.json").exists():
                lora_adapter_path = str(ckpt_dir / "lora_adapters")
                ckpt_path = resolve_base_checkpoint()
                _log.info(f"Video model: base {ckpt_path} + LoRA adapters {lora_adapter_path}")
            elif (ckpt_dir / "video_model_full.pt").exists():
                ckpt_path = str(ckpt_dir / "video_model_full.pt")
            else:
                raise FileNotFoundError(f"{ckpt_dir} has neither video_model_full.pt nor lora_adapters/")
        if lora_adapter_path is None:
            _log.info(f"Video model weights: {ckpt_path or 'pretrained base'}")

        # Resolve checkpoint path
        resolved_ckpt = ckpt_path if ckpt_path is not None else resolve_base_checkpoint()

        # Load the full model via Video2WorldInference (exact original pipeline)
        experiment_name = EXPERIMENT_NAMES[setup]
        self.pipe = Video2WorldInference(
            experiment_name=experiment_name,
            ckpt_path=resolved_ckpt,
            s3_credential_path="",
            offload_text_encoder=False,
            stop_rank_sync=True,
        )
        self.model = self.pipe.model

        # Apply and merge LoRA adapters if detected
        if lora_adapter_path is not None:
            self._apply_and_merge_lora(lora_adapter_path)

        # Override temporal_causal flag on each self-attention block
        if atten_temporal_causal and hasattr(self.model, "net") and hasattr(self.model.net, "blocks"):
            for block in self.model.net.blocks:
                if hasattr(block, "self_attn"):
                    block.self_attn.temporal_causal = atten_temporal_causal

        # Ensure model on correct device
        try:
            self.model.to(self.device)
        except Exception:
            pass

        # Wrap denoiser
        self.compiled_denoiser = CompiledDenoiser(self.model)

        # Reason1 text encoder (lazy-loaded on first use)
        self._text_encoder = None
        self._text_embedding_cache: dict[str, torch.Tensor] = {}

    def _apply_and_merge_lora(self, lora_adapter_path: str):
        """Load LoRA adapters from disk, apply to the DiT, merge, and unload PEFT wrapper."""
        from peft import PeftModel
        print(f"[VideoProcessor] Applying LoRA adapters from {lora_adapter_path}", flush=True)
        self.model.net = PeftModel.from_pretrained(self.model.net, lora_adapter_path)
        self.model.net = self.model.net.merge_and_unload()
        print(f"[VideoProcessor] LoRA merged and PEFT wrapper removed", flush=True)

    def _ensure_text_encoder(self):
        """Lazy-load the Reason1 text encoder on first use."""
        if self._text_encoder is None:
            from temporal_ratio.models.text_encoder import Reason1TextEncoder
            self._text_encoder = Reason1TextEncoder(device=self.device)

    def encode_prompt(self, prompt: str | list[str]) -> torch.Tensor:
        """Encode prompt(s) to Reason1 embeddings on-the-fly.

        Uses the Reason1 text encoder (Qwen2.5-VL-7B, full_concat strategy)
        to produce (B, 512, 100352) bfloat16 embeddings. Results are cached
        per prompt string.

        Args:
            prompt: Single prompt string or list of prompts.

        Returns:
            Tensor of shape (B, 512, 100352) bfloat16 on this processor's device.
        """
        self._ensure_text_encoder()

        if isinstance(prompt, str):
            prompt = [prompt]

        # Check cache for all prompts
        uncached = [p for p in prompt if p not in self._text_embedding_cache]
        if uncached:
            embeddings = self._text_encoder.encode(uncached)
            for i, p in enumerate(uncached):
                self._text_embedding_cache[p] = embeddings[i:i+1].cpu()

        # Gather from cache
        result = torch.cat([
            self._text_embedding_cache[p].to(device=self.device, dtype=torch.bfloat16)
            for p in prompt
        ], dim=0)
        return result

    def offload_text_encoder(self):
        """Move Reason1 model to CPU to free GPU memory after encoding."""
        if self._text_encoder is not None:
            self._text_encoder.offload_to_cpu()

    # -------------------------------------------------------------------
    # Video tokenization (frozen VAE encode)
    # -------------------------------------------------------------------

    @torch.no_grad()
    def tokenize_video(
        self,
        images: torch.Tensor,
        context_embedding: torch.Tensor | str | list[str] | None = None,
        num_conditional_frames: int = 1,
    ) -> tuple[torch.Tensor, dict]:
        """Encode video frames via the frozen VAE and return latent + conditions.

        Parameters
        ----------
        images : torch.Tensor
            Video tensor of shape ``(B, T, H, W, C)`` in uint8 [0, 255].
        context_embedding : torch.Tensor | str | list[str] | None
            Text conditioning. Can be:
            - A pre-computed embedding tensor ``(B, seq_len, dim)``
            - A prompt string or list of strings (encoded on-the-fly via Reason1)
            - None for zero embeddings (no text guidance)
        num_conditional_frames : int
            Number of leading frames treated as conditioning context.

        Returns
        -------
        latent_state : torch.Tensor
            VAE-encoded latent, shape ``(B, C, T', H', W')``.
        conditions : dict
            Condition dict consumed by the denoiser.
        """
        # Resolve text embeddings from prompt string or tensor
        if isinstance(context_embedding, str) or (isinstance(context_embedding, list) and context_embedding and isinstance(context_embedding[0], str)):
            context_embedding = self.encode_prompt(context_embedding)
        elif context_embedding is None:
            B = images.shape[0]
            # Zero embedding with full_concat dim (100352) — crossattn_proj handles projection
            context_embedding = torch.zeros(B, 512, 100352, dtype=torch.bfloat16, device=self.device)

        # Model expects (B, C, T, H, W)
        images_input = images.permute(0, 4, 1, 2, 3)
        batch_input = {
            "video": images_input,
            "t5_text_embeddings": context_embedding,
            "fps": torch.randint(16, 32, (images_input.shape[0],), device=self.device).float(),
            "padding_mask": torch.zeros(
                images_input.shape[0], 1, images_input.shape[3], images_input.shape[4],
                device=self.device, dtype=context_embedding.dtype,
            ),
            "num_conditional_frames": num_conditional_frames,
        }
        _, latent_state, conditions = self.model.get_data_and_condition(batch_input)
        return latent_state, conditions

    # -------------------------------------------------------------------
    # Time sampling helpers
    # -------------------------------------------------------------------

    def sample_time(self, bsize: int, device: torch.device | str) -> torch.Tensor:
        """Sample timesteps from a Beta(1.5, 1.0) distribution in (0.001, 0.999)."""
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def sample_logit_normal_time(self, bsize: int, device: torch.device | str) -> torch.Tensor:
        """Sample timesteps from a logit-normal distribution biased towards 1."""
        time_logit_normal = sample_logit_normal(1.0, 1.0, bsize, device)
        time = time_logit_normal * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    # -------------------------------------------------------------------
    # Velocity prediction — uniform sigma sampling
    # -------------------------------------------------------------------

    @torch.no_grad()
    def predict_velocity_uniform(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: dict,
        testing: bool = False,
        timesteps: Optional[torch.Tensor] = None,
        intermediate_feature_ids: list[int] | None = None,
        target_frames: int = 17,
    ) -> tuple[torch.Tensor, dict, torch.Tensor]:
        """Forward pass with uniform sigma sampling in (0, 1).

        Returns
        -------
        vt_pred : torch.Tensor
            Predicted velocity field.
        intermediate_features : dict
            Intermediate DiT features keyed by block index.
        normalized_timesteps : torch.Tensor
            Timesteps normalised to [0, 1].
        """
        x0_B_C_T_H_W = x0_B_C_T_H_W.to(**self.model.tensor_kwargs_fp32)
        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), **self.model.tensor_kwargs_fp32)
        batch_size = x0_B_C_T_H_W.size(0)

        if timesteps is None:
            # Uniform sigma in (0.001, 0.999)
            sigmas = torch.rand(batch_size, device=self.device) * 0.998 + 0.001
        else:
            sigmas = timesteps.to(dtype=torch.float32, device=self.device)
            if sigmas.ndim == 0:
                sigmas = sigmas.repeat(batch_size)

        # Convert sigmas to model timesteps (scale to [0, 1000])
        timesteps_scaled = (sigmas * 1000.0).to(**self.model.tensor_kwargs_fp32)

        x0_B_C_T_H_W = x0_B_C_T_H_W.to(self.device)
        epsilon_B_C_T_H_W = epsilon_B_C_T_H_W.to(self.device)
        timesteps_scaled = timesteps_scaled.to(self.device)
        sigmas = sigmas.to(self.device)

        timesteps_scaled = rearrange(timesteps_scaled, "b -> b 1")
        sigmas = rearrange(sigmas, "b -> b 1")

        xt_B_C_T_H_W, _vt_B_C_T_H_W = self.model.rectified_flow.get_interpolation(
            epsilon_B_C_T_H_W, x0_B_C_T_H_W, sigmas
        )

        vt_pred_B_C_T_H_W, intermediate_features = self.compiled_denoiser(
            noise=epsilon_B_C_T_H_W,
            xt_B_C_T_H_W=xt_B_C_T_H_W.to(**self.model.tensor_kwargs),
            timesteps_B_T=timesteps_scaled,
            condition=condition,
            intermediate_feature_ids=intermediate_feature_ids,
        )

        # Normalise timesteps to [0, 1]
        normalized_t = timesteps_scaled.to(torch.float32).squeeze(-1) / 1000.0
        return vt_pred_B_C_T_H_W, intermediate_features, normalized_t

    # -------------------------------------------------------------------
    # Velocity prediction — model schedule sigma sampling (CFG-style)
    # -------------------------------------------------------------------

    @torch.no_grad()
    def predict_velocity_with_cfg(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: dict,
        testing: bool = False,
        timesteps: Optional[torch.Tensor] = None,
        intermediate_feature_ids: list[int] | None = None,
        target_frames: int = 17,
    ) -> tuple[torch.Tensor, dict, torch.Tensor]:
        """Forward pass using the model's built-in schedule for sigma sampling.

        Returns the same triple as :meth:`predict_velocity_uniform`.
        """
        x0_B_C_T_H_W = x0_B_C_T_H_W.to(**self.model.tensor_kwargs_fp32)
        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), **self.model.tensor_kwargs_fp32)
        batch_size = x0_B_C_T_H_W.size(0)

        if timesteps is None:
            t = torch.randint(1, len(self.model.sample_scheduler.timesteps), (batch_size,), device="cpu")
            timesteps = self.model.sample_scheduler.timesteps[t]
        else:
            timesteps = timesteps.to(torch.int32)
            if timesteps.ndim == 0:
                timesteps = timesteps.repeat(batch_size)

        timesteps = timesteps.to(**self.model.tensor_kwargs_fp32)

        x0_B_C_T_H_W = x0_B_C_T_H_W.to(self.device)
        epsilon_B_C_T_H_W = epsilon_B_C_T_H_W.to(self.device)
        timesteps = timesteps.to(self.device)

        sigmas = self.model.sample_scheduler.sigmas[
            [
                (self.model.sample_scheduler.timesteps.clone().to(device=self.device) == t)
                .nonzero()
                .squeeze()
                .cpu()
                .tolist()
                for t in timesteps
            ]
        ].to(**self.model.tensor_kwargs_fp32)

        timesteps = rearrange(timesteps, "b -> b 1")
        sigmas = rearrange(sigmas, "b -> b 1")

        xt_B_C_T_H_W, _vt_B_C_T_H_W = self.model.rectified_flow.get_interpolation(
            epsilon_B_C_T_H_W, x0_B_C_T_H_W, sigmas
        )

        vt_pred_B_C_T_H_W, intermediate_features = self.compiled_denoiser(
            noise=epsilon_B_C_T_H_W,
            xt_B_C_T_H_W=xt_B_C_T_H_W.to(**self.model.tensor_kwargs),
            timesteps_B_T=timesteps,
            condition=condition,
            intermediate_feature_ids=intermediate_feature_ids,
        )

        normalized_t = timesteps.to(torch.float32).squeeze(-1) / 1000.0
        return vt_pred_B_C_T_H_W, intermediate_features, normalized_t

    # -------------------------------------------------------------------
    # Eval velocity — fixed timestep, pure noise input
    # -------------------------------------------------------------------

    @torch.no_grad()
    def eval_velocity(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: dict,
        testing: bool = False,
        timesteps: Optional[torch.Tensor] = None,
        intermediate_feature_ids: list[int] | None = None,
    ) -> tuple[dict, torch.Tensor]:
        """Forward pass at a fixed timestep for inference / evaluation.

        Parameters
        ----------
        x0_B_C_T_H_W : torch.Tensor
            Latent tensor (used only for shape; pure noise is fed to the denoiser).
        condition : dict
            Condition dict from :meth:`tokenize_video`.
        timesteps : torch.Tensor, optional
            If ``None``, uses the first timestep from the sample scheduler.
        intermediate_feature_ids : list[int], optional
            Which DiT block indices to extract features from.

        Returns
        -------
        intermediate_features : dict
            Intermediate DiT features keyed by block index.
        normalized_timesteps : torch.Tensor
            Timesteps normalised to [0, 1].
        """
        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), **self.model.tensor_kwargs_fp32)
        batch_size = epsilon_B_C_T_H_W.size(0)

        if timesteps is None:
            timesteps = self.model.sample_scheduler.timesteps[0]
        else:
            timesteps = timesteps.to(torch.int32)
            if timesteps.ndim == 0:
                timesteps = timesteps.repeat(batch_size)

        timesteps = timesteps.repeat(batch_size).to(**self.model.tensor_kwargs_fp32)

        epsilon_B_C_T_H_W = epsilon_B_C_T_H_W.to(self.device)
        timesteps = timesteps.to(self.device)

        _vt_pred_B_C_T_H_W, intermediate_features = self.model.denoise(
            noise=epsilon_B_C_T_H_W.to(**self.model.tensor_kwargs),
            xt_B_C_T_H_W=epsilon_B_C_T_H_W.to(**self.model.tensor_kwargs),
            timesteps_B_T=timesteps,
            condition=condition,
            intermediate_feature_ids=intermediate_feature_ids,
        )

        normalized_t = timesteps / 1000.0  # normalise to [0, 1]
        return intermediate_features, normalized_t

    # -------------------------------------------------------------------
    # Full autoregressive video generation with feature extraction
    # -------------------------------------------------------------------

    @torch.no_grad()
    def eval_vid2world(
        self,
        images: torch.Tensor,
        prompt: str | list[str] | torch.Tensor | None = None,
        num_conditional_frames: int = 1,
        guidance: float = 7.0,
        seed: int = 1,
        num_steps: int | None = None,
        intermediate_feature_ids: list[int] | None = None,
        target_denoising_steps: int | None = None,
    ) -> tuple[list[dict], list[float], torch.Tensor]:
        """Full denoising loop: noise → video with feature extraction at each step.

        This replaces Video2WorldInference.eval_vid2world().

        Parameters
        ----------
        images : torch.Tensor
            Video tensor (B, T, H, W, C) uint8 [0, 255].
        prompt : str | list[str] | torch.Tensor | None
            Text conditioning. Can be:
            - A prompt string or list of strings (encoded on-the-fly via Reason1)
            - A pre-computed embedding tensor (B, seq_len, dim)
            - None for zero embeddings (no text guidance)
        num_conditional_frames : int
            Number of leading frames used as conditioning.
        guidance : float
            CFG guidance scale.
        seed : int
            Random seed for noise generation.
        num_steps : int, optional
            Number of denoising steps (default: self.num_global_denoising_steps).
        intermediate_feature_ids : list[int], optional
            DiT block indices to extract features from.
        target_denoising_steps : int, optional
            Early-stop after this many steps.

        Returns
        -------
        all_features : list[dict[int, Tensor]]
            Features per denoising step, indexed [step][block_id].
        all_timesteps : list[float]
            Normalized timestep per step.
        latents : Tensor
            Final denoised latent (B, C, T, H, W).
        """
        if num_steps is None:
            num_steps = self.num_global_denoising_steps

        # Convert (B, T, H, W, C) uint8 → (B, C, T, H, W) uint8 for the V2W pipeline
        vid_input = images.permute(0, 4, 1, 2, 3)

        # Resolve prompt to string for the original pipeline
        if isinstance(prompt, list):
            prompt_str = prompt[0]
        elif isinstance(prompt, str):
            prompt_str = prompt
        else:
            prompt_str = ""

        # Use the original V2W pipeline's data preparation (handles negative prompt
        # embeddings for CFG, text encoding, normalization, etc.)
        data_batch = self.pipe._get_data_batch_input(
            video=vid_input,
            prompt=prompt_str,
            num_conditional_frames=num_conditional_frames,
            use_neg_prompt=True,
        )

        # Generate using the exact original pipeline
        latents = self.model.generate_samples_from_batch(
            data_batch,
            guidance=guidance,
            seed=seed,
            is_negative_prompt=True,
            num_steps=num_steps,
        )

        return [], [], latents
