"""Temporal Ratio (TR) from action-head attention (numpy only; unit-tested on CPU).

The action head attends to the video latent tokens of ``T`` latent frames: frame 0 is the
anchored current observation, frames 1..T-1 are predicted future frames. For the action
queries' head-averaged attention ``A`` (summed over the tokens of each frame),

    TR = sum_{t>=1} A[t] / A[0]

TR > 1 means the action head puts more attention mass on the predicted future than on the
current frame. Note: TR is not normalized by token count, so uniform attention gives TR = T-1.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

EPS = 1e-8


def temporal_ratio(frame_attention: Sequence[float] | np.ndarray) -> float:
    """TR from per-latent-frame attention mass (any length >= 2)."""
    a = np.asarray(frame_attention, dtype=np.float64)
    if a.ndim != 1 or a.shape[0] < 2:
        raise ValueError(f"need attention mass for >= 2 latent frames, got shape {a.shape}")
    return float(a[1:].sum() / (a[0] + EPS))


def frame_attention_from_tokens(token_attention: np.ndarray, tokens_per_frame: int) -> np.ndarray:
    """Sum attention over each latent frame's tokens.

    Args:
        token_attention: ``(..., T * tokens_per_frame)`` attention from action queries to video tokens
            (already averaged over heads / action queries as desired).
    """
    a = np.asarray(token_attention)
    if a.shape[-1] % tokens_per_frame:
        raise ValueError(f"{a.shape[-1]} video tokens is not a multiple of {tokens_per_frame}")
    return a.reshape(*a.shape[:-1], -1, tokens_per_frame).sum(-1)


def temporal_ratio_from_segments(segment_attention: np.ndarray, segment_names: Sequence[str], layer: int = 0) -> float:
    """TR from ``visualize_attention`` output: ``(layers, actions, segments)`` + segment names."""
    seg = np.asarray(segment_attention)
    video = [i for i, n in enumerate(segment_names) if n.startswith("video_t")]
    if len(video) < 2:
        return 1.0
    per_seg = seg.mean(axis=1)[layer]
    return temporal_ratio(per_seg[video])
