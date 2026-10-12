"""PyAV frame decoding for LeRobot datasets.

LeRobot 0.1.0 decodes dataset videos either with ``torchcodec`` or, for its ``pyav`` backend,
through ``torchvision.io.VideoReader``, which recent torchvision releases (>= 0.26, required by
CUDA 13 builds of torch) no longer provide. ``install()`` routes LeRobot's ``pyav`` backend to
an equivalent decoder built directly on PyAV; it is a no-op when ``VideoReader`` exists.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

log = logging.getLogger(__name__)


def decode_video_frames_pyav(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Frames closest to ``timestamps`` (seconds) as float32 ``(T, C, H, W)`` in [0, 1].

    Same contract as ``lerobot.common.datasets.video_utils.decode_video_frames_torchvision``:
    seek to the key frame preceding the first timestamp, decode up to the last one and pick the
    closest decoded frame for every query (which must lie within ``tolerance_s``).
    """
    import av

    first_ts, last_ts = min(timestamps), max(timestamps)
    loaded_frames: list[torch.Tensor] = []
    loaded_ts: list[float] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        container.seek(int(first_ts / stream.time_base), stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            ts = float(frame.time)
            if log_loaded_timestamps:
                log.info(f"frame loaded at timestamp={ts:.4f}")
            loaded_frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1))
            loaded_ts.append(ts)
            if ts >= last_ts:
                break

    query = torch.tensor(timestamps)
    loaded = torch.tensor(loaded_ts)
    dist = torch.cdist(query[:, None], loaded[:, None], p=1)
    min_, argmin_ = dist.min(1)
    if not (min_ < tolerance_s).all():
        raise AssertionError(
            f"query timestamps {query[min_ >= tolerance_s].tolist()} are further than {tolerance_s=} s from "
            f"any decoded frame of {video_path} (loaded {loaded.tolist()[:4]}...)"
        )
    frames = torch.stack([loaded_frames[i] for i in argmin_])
    return frames.to(torch.float32) / 255


def install() -> None:
    """Use :func:`decode_video_frames_pyav` for LeRobot's ``pyav`` backend if torchvision lacks it."""
    import torchvision

    if hasattr(torchvision.io, "VideoReader"):
        return
    from lerobot.common.datasets import video_utils

    if video_utils.decode_video_frames_torchvision is not decode_video_frames_pyav:
        video_utils.decode_video_frames_torchvision = decode_video_frames_pyav
        log.info("LeRobot video decoding: PyAV (torchvision.io.VideoReader unavailable)")
