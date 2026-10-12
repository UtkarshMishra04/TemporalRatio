"""Frozen pretrained encoders (DINOv2, CLIP) for REPA alignment targets.

Extracts patch-level features from single images. Always frozen, no gradients.
Used by JointVideoActionModel to compute REPA goal-guidance loss.

Multi-GPU safe: only local rank 0 downloads; other ranks wait then load from cache.
"""

import logging
import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

log = logging.getLogger(__name__)

ENCODER_CONFIGS = {
    "dinov2-vit-b": {"hub": "facebookresearch/dinov2", "model": "dinov2_vitb14", "dim": 768, "patch": 14},
    "dinov2-vit-l": {"hub": "facebookresearch/dinov2", "model": "dinov2_vitl14", "dim": 1024, "patch": 14},
    "dinov2-vit-g": {"hub": "facebookresearch/dinov2", "model": "dinov2_vitg14", "dim": 1536, "patch": 14},
    "clip-vit-l": {"dim": 1024, "patch": 14},
}


def _is_local_rank_zero() -> bool:
    """Check if this is local rank 0 (safe to download)."""
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) == 0


def _barrier_if_distributed():
    """Barrier if DDP is initialized."""
    if dist.is_initialized():
        dist.barrier()


class REPAEncoder(nn.Module):
    """Frozen pretrained encoder for REPA alignment targets.

    Supports DINOv2 (B/L/G) and CLIP (ViT-L/14). Extracts patch tokens
    from a single image, always in eval mode with no gradients.

    Multi-GPU: only rank 0 downloads; others wait then load from cache.
    """

    def __init__(self, encoder_name: str = "dinov2-vit-b", device: str | torch.device = "cuda"):
        super().__init__()
        if encoder_name not in ENCODER_CONFIGS:
            raise ValueError(f"Unknown encoder: {encoder_name}. Choose from {list(ENCODER_CONFIGS.keys())}")

        self.encoder_name = encoder_name
        self.config = ENCODER_CONFIGS[encoder_name]
        self.output_dim = self.config["dim"]
        self.patch_size = self.config["patch"]
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # Rank-safe loading: rank 0 downloads, barrier, then all ranks load from cache
        is_rank0 = _is_local_rank_zero()

        if is_rank0:
            log.info(f"Rank 0: downloading/loading {encoder_name}...")
            self.encoder = self._load_encoder(encoder_name)
        _barrier_if_distributed()
        if not is_rank0:
            log.info(f"Rank {os.environ.get('LOCAL_RANK', '?')}: loading {encoder_name} from cache...")
            self.encoder = self._load_encoder(encoder_name)

        self.encoder.to(device=self.device, dtype=torch.float16)
        self.encoder.eval()

        # Freeze everything
        for param in self.encoder.parameters():
            param.requires_grad = False

        num_params = sum(p.numel() for p in self.encoder.parameters()) / 1e6
        log.info(f"REPAEncoder: {encoder_name}, dim={self.output_dim}, patch={self.patch_size}, "
                 f"params={num_params:.0f}M, device={self.device}")

    def _load_encoder(self, encoder_name: str):
        """Load encoder model (from cache if available)."""
        if encoder_name.startswith("dinov2"):
            return torch.hub.load(
                self.config["hub"], self.config["model"], pretrained=True
            )
        elif encoder_name == "clip-vit-l":
            return self._load_clip()
        else:
            raise ValueError(f"Unsupported encoder: {encoder_name}")

    def _load_clip(self):
        """Load CLIP ViT-L/14 vision encoder."""
        try:
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="openai"
            )
            return model.visual
        except ImportError:
            from transformers import CLIPVisionModel
            model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
            return model

    @torch.no_grad()
    def forward(self, images_BHWC: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens from images.

        Args:
            images_BHWC: (B, H, W, 3) uint8 [0,255] or float [0,1] tensor.

        Returns:
            features: (B, num_patches, encoder_dim) float tensor.
                num_patches = (H // patch_size) * (W // patch_size)
        """
        # Convert to float [0, 1] if uint8
        x = images_BHWC.float()
        if x.max() > 1.0:
            x = x / 255.0

        # (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # ImageNet normalize
        x = self.normalize(x)

        x = x.to(device=self.device, dtype=torch.float16)

        if self.encoder_name.startswith("dinov2"):
            return self._forward_dinov2(x)
        elif self.encoder_name == "clip-vit-l":
            return self._forward_clip(x)
        else:
            raise ValueError(f"Unsupported encoder: {self.encoder_name}")

    def _forward_dinov2(self, x_BCHW: torch.Tensor) -> torch.Tensor:
        """DINOv2 forward — returns patch tokens (no CLS)."""
        features = self.encoder.forward_features(x_BCHW)
        # DINOv2 returns dict with 'x_norm_patchtokens' key
        if isinstance(features, dict):
            tokens = features["x_norm_patchtokens"]  # (B, num_patches, dim)
        else:
            # Fallback: remove CLS token
            tokens = features[:, 1:, :]
        return tokens.float()

    def _forward_clip(self, x_BCHW: torch.Tensor) -> torch.Tensor:
        """CLIP forward — returns patch tokens (no CLS).

        CLIP ViT-L/14 expects 224x224 input. Non-square images are resized.
        The spatial mismatch with DiT features is handled by bilinear
        interpolation in compute_repa_loss.
        """
        # CLIP requires 224x224 — resize if needed
        if x_BCHW.shape[2] != 224 or x_BCHW.shape[3] != 224:
            x_BCHW = F.interpolate(x_BCHW, size=(224, 224), mode="bilinear", align_corners=False)

        try:
            # open_clip path
            features = self.encoder.forward(x_BCHW)
            if isinstance(features, tuple):
                features = features[0]
            # Remove CLS if present
            expected_patches = (224 // self.patch_size) ** 2  # 256 for patch=14
            if features.shape[1] > expected_patches:
                features = features[:, 1:, :]
            return features.float()
        except Exception:
            # transformers CLIPVisionModel path
            outputs = self.encoder(pixel_values=x_BCHW)
            # last_hidden_state includes CLS at position 0
            tokens = outputs.last_hidden_state[:, 1:, :]
            return tokens.float()
