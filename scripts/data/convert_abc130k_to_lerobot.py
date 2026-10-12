#!/usr/bin/env python3
"""Convert ABC-130k (XDOF/ABC-130k, MCAP) episodes into the LeRobot layout of the YAM checkpoints.

The bimanual YAM checkpoints of this repository were trained on LeRobot **v2.1** datasets
("pp24", 24 tasks; 22 are in ABC-130k, see ``temporal_ratio.data.abc130k``) with

* ``observation.state`` / ``action``: float32[14] =
  ``[left joint1..6 (rad), left gripper, right joint1..6 (rad), right gripper]``
  with the gripper as a normalized aperture in [0, 1] (0 = closed, 1 = open),
  ``action`` being the *commanded* (teleop) joint/gripper positions,
* three AV1 (yuv420p) videos at 168x224 (HxW), 30 fps:
  ``observation.images.left_image`` (top camera),
  ``observation.images.left_wrist_left``, ``observation.images.right_wrist_left``,
* ``next.done``, ``timestamp``, ``frame_index``, ``episode_index``, ``index``,
  ``task_index`` columns, and a per-episode prompt resolved through
  ``meta/tasks.jsonl`` (``prompt_from_task=True`` in the training config).

This script reads ABC-130k ``episode.mcap`` files (RealSense or ZED-X
stations), resamples every stream onto a uniform ``--fps`` grid, resizes the
cameras to ``--image-size`` and writes that layout (parquet + mp4 + meta
json/jsonl) without depending on ``lerobot``; parquet files are written the
way ``lerobot`` 0cf8648 + ``datasets<3`` (the training stack) writes them.
See docs/data.md for the schema and conversion details.

Examples::

    # straight from the Hub: 50 episodes of a task -> $HF_LEROBOT_HOME/abc130k/<task>
    python scripts/data/convert_abc130k_to_lerobot.py --src hf:fold_and_stack_the_t_shirts --max-episodes 50

    # a whole task, downloading inside the workers and deleting each raw episode once
    # converted; rerun with --resume after an interruption
    python scripts/data/convert_abc130k_to_lerobot.py --src hf:load_the_cups_into_the_dish_rack \\
        --workers 24 --delete-raw

    # local episodes (any directory containing */episode.mcap)
    python scripts/data/convert_abc130k_to_lerobot.py --src /data/abc130k/train/fold_and_stack_the_t_shirts
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

log = logging.getLogger("abc130k_to_lerobot")

# --------------------------------------------------------------------------------------
# pp24 schema constants
# --------------------------------------------------------------------------------------

ABC_REPO_ID = "XDOF/ABC-130k"
CODEBASE_VERSION = "v2.1"
STATE_DIM = 14
ACTION_DIM = 14
DEFAULT_CHUNKS_SIZE = 1000
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
MANIFEST_NAME = "abc130k_conversion.json"
DATASETS_CDC_OPTIONS = {"min_chunk_size": 256 * 1024, "max_chunk_size": 1024 * 1024, "norm_level": 0}  # datasets>=4
DATASETS3_BATCH_SIZE = 1000  # datasets<4 writer batch (= row group) size

# pp24 camera keys, in the order the training code tiles them horizontally
# ([left_image | left_wrist_left | right_wrist_left]).
CAM_TOP = "observation.images.left_image"
CAM_LEFT_WRIST = "observation.images.left_wrist_left"
CAM_RIGHT_WRIST = "observation.images.right_wrist_left"
CAMERA_KEYS: tuple[str, ...] = (CAM_TOP, CAM_LEFT_WRIST, CAM_RIGHT_WRIST)

# (topic, number of values taken from RobotState/GripperState.position), concatenated in this
# order. pp24 dims: 0-5 left joints, 6 left gripper, 7-12 right joints, 13 right gripper.
STATE_TOPICS: tuple[tuple[str, int], ...] = (
    ("/left-arm-state", 6),
    ("/left-ee-state", 1),
    ("/right-arm-state", 6),
    ("/right-ee-state", 1),
)
ACTION_TOPICS: tuple[tuple[str, int], ...] = (
    ("/left-arm-action", 6),
    ("/left-ee-action", 1),
    ("/right-arm-action", 6),
    ("/right-ee-action", 1),
)
GRIPPER_DIMS = (6, 13)
ROBOT_TOPICS = tuple(t for t, _ in STATE_TOPICS + ACTION_TOPICS)

# ABC-130k task directory -> (pp24 skill, pp24 task). 22 of the 24 pp24 tasks have an ABC
# counterpart; pickAndPlace.handoff and pickAndPlace.putRelative do not.
ABC_TO_PP24: dict[str, tuple[str, str]] = {
    "fold_and_stack_the_t_shirts": ("folding", "foldingTshirtPileAndStacking"),
    "remove_the_shirt_from_the_hanger": ("garmentManipulation", "removeTheShirtFromTheHanger"),
    "remove_the_t_shirt_from_the_hanger": ("garmentManipulation", "removeTheTshirtFromTheHanger"),
    "unscrew_the_bottle_caps": ("highPrecisionAssembly", "unscrewBottleCaps"),
    "place_the_mixed_dishes_into_the_plastic_bin": ("pickAndPlace", "busTablePlacingMixedDishesGlassesIntoAPlasticBin"),
    "place_the_plates_into_the_plastic_bin_on_the_countertop": ("pickAndPlace", "busTablePlacingPlatesIntoAPlasticBin"),
    "load_the_bowls_into_the_dish_rack": ("pickAndPlace", "loadBowlsIntoTabletopDishRack"),
    "load_the_cups_into_the_dish_rack": ("pickAndPlace", "loadCupsIntoTabletopDishRack"),
    "load_the_mixed_dishes_into_the_dish_rack": ("pickAndPlace", "loadMixedDishesIntoTabletopDishRack"),
    "load_the_plates_into_the_dish_rack": ("pickAndPlace", "loadPlatesIntoTabletopDishRack"),
    "place_and_organize_the_beverage_onto_the_shelf": ("pickAndPlace", "placeOrganizeBeverageOntoShelf"),
    "place_and_organize_the_canned_foods_onto_the_counter": ("pickAndPlace", "placeOrganizeCannedFoodsOntoCounter"),
    "place_and_organize_the_chips_bags_onto_the_shelf": ("pickAndPlace", "placeOrganizeChipsBagsOntoShelf"),
    "place_and_organize_the_fake_fruits_in_the_fruit_bowl": ("pickAndPlace", "placeOrganizeFakeFruitsIntoFruitBowl"),
    "place_the_snacks_into_the_paper_bag": ("pickAndPlace", "placeSnacksIntoPaperBag"),
    "take_the_snacks_out_of_the_paper_bag": ("pickAndPlace", "takeSnacksOutOfPaperBag"),
    "throw_the_plastic_bottles_in_the_bin": ("pickAndPlace", "throwPlasticBottlesInBin"),
    "turn_the_mug_right_side_up": ("pickAndPlace", "turnMugRightSideUp"),
    "unload_the_bowls_from_the_dish_rack": ("pickAndPlace", "unloadBowlsFromTabletopDishRack"),
    "unload_the_cups_from_the_dish_rack": ("pickAndPlace", "unloadCupsFromTabletopDishRack"),
    "unload_the_mixed_dishes_from_the_dish_rack": ("pickAndPlace", "unloadMixedDishesFromTabletopDishRack"),
    "unload_the_plates_from_the_dish_rack": ("pickAndPlace", "unloadPlatesFromTabletopDishRack"),
}

_CODEC_FOR_FORMAT = {"h264": "h264", "avc": "h264", "h265": "hevc", "hevc": "hevc"}


def pp24_features(height: int, width: int, video_info: dict[str, Any]) -> dict[str, Any]:
    """Feature dict of meta/info.json, in pp24 key order."""
    video_feature = {
        "dtype": "video",
        "shape": [height, width, 3],
        "names": ["height", "width", "channel"],
        "info": dict(video_info),
    }
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": [STATE_DIM],
            "names": [f"state_{i}" for i in range(STATE_DIM)],
        },
        "action": {
            "dtype": "float32",
            "shape": [ACTION_DIM],
            "names": [f"action_{i}" for i in range(ACTION_DIM)],
        },
        "next.done": {"dtype": "bool", "shape": [1], "names": ["done"]},
        **{key: json.loads(json.dumps(video_feature)) for key in CAMERA_KEYS},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }


def hf_features_metadata(style: str) -> str | None:
    """`huggingface` parquet schema metadata as written by `datasets` (style 'list' = datasets>=4)."""
    if style == "none":
        return None

    def seq(n: int) -> dict[str, Any]:
        if style == "list":
            return {"feature": {"dtype": "float32", "_type": "Value"}, "length": n, "_type": "List"}
        return {"feature": {"dtype": "float32", "_type": "Value"}, "length": n, "_type": "Sequence"}

    feats = {
        "observation.state": seq(STATE_DIM),
        "action": seq(ACTION_DIM),
        "next.done": {"dtype": "bool", "_type": "Value"},
        "timestamp": {"dtype": "float32", "_type": "Value"},
        "frame_index": {"dtype": "int64", "_type": "Value"},
        "episode_index": {"dtype": "int64", "_type": "Value"},
        "index": {"dtype": "int64", "_type": "Value"},
        "task_index": {"dtype": "int64", "_type": "Value"},
    }
    return json.dumps({"info": {"features": feats}})


# --------------------------------------------------------------------------------------
# Configuration / data containers
# --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ConvertConfig:
    """Per-episode conversion settings (picklable, shipped to worker processes)."""

    fps: int = 30
    height: int = 168
    width: int = 224
    aspect: str = "crop"  # crop | stretch | pad, for sources that are not height:width
    resize_filter: str = "area"
    top_eye: str = "left"  # ZED-X stereo top camera eye used for left_image
    align: str = "nearest"  # nearest | linear (robot streams; cameras are always nearest)
    camera_offset_ms: float = 0.0
    invert_gripper: bool = False
    max_frames: int | None = None
    min_frames: int = 2
    vcodec: str = "libsvtav1"
    pix_fmt: str = "yuv420p"
    crf: int | None = 30
    gop: int | None = 2
    svt_preset: int | None = None
    decoder_threads: int = 4
    prompt_source: str = "instruction"  # instruction | metadata | task-dir
    delete_raw: bool = False  # delete Hub-downloaded MCAPs once the episode is converted


@dataclasses.dataclass(frozen=True)
class EpisodeSource:
    """One ABC-130k episode, on local disk or (hub_files set) downloaded lazily by the worker."""

    mcap: Path
    annotation: Path | None
    task_dir: str
    split: str | None
    episode_id: str
    hub_files: tuple[str, ...] = ()  # repo paths to fetch into download_dir before converting
    download_dir: str | None = None
    hub_bytes: int = 0

    def fetch(self) -> None:
        """Download the episode's MCAP files from the Hub if they are not on disk yet."""
        if not self.hub_files or (self.mcap.exists() and (self.annotation is None or self.annotation.exists())):
            return
        from huggingface_hub import hf_hub_download

        for path in self.hub_files:
            hf_hub_download(ABC_REPO_ID, path, repo_type="dataset", local_dir=self.download_dir)

    def delete_download(self) -> None:
        """Remove a Hub-sourced episode directory (and its huggingface_hub local_dir metadata)."""
        if not self.hub_files or self.download_dir is None:
            return
        shutil.rmtree(self.mcap.parent, ignore_errors=True)
        rel = self.mcap.parent.relative_to(self.download_dir)
        shutil.rmtree(Path(self.download_dir) / ".cache" / "huggingface" / "download" / rel, ignore_errors=True)

    @classmethod
    def from_mcap(cls, mcap: Path) -> "EpisodeSource":
        ep_dir = mcap.parent
        ann = ep_dir / "annotation.mcap"
        episode_id = ep_dir.name[len("episode_"):] if ep_dir.name.startswith("episode_") else ep_dir.name
        task_dir = ep_dir.parent.name
        if task_dir.startswith("task="):  # layout described on the dataset card
            task_dir = task_dir[len("task="):]
        split = ep_dir.parent.parent.name if ep_dir.parent.parent.name in {"train", "val", "eval", "dagger", "sim"} else None
        return cls(mcap=mcap, annotation=ann if ann.exists() else None, task_dir=task_dir, split=split, episode_id=episode_id)


@dataclasses.dataclass
class CameraStream:
    topic: str
    codec: str
    timestamps: np.ndarray  # int64 ns, message order (== decode order)
    packets: list[bytes]
    calib: dict[str, Any] | None = None


@dataclasses.dataclass
class RawEpisode:
    metadata: dict[str, str]
    instruction: str | None
    station: str
    robot: dict[str, tuple[np.ndarray, np.ndarray]]  # topic -> (ts int64 [M], values float64 [M, D])
    cameras: dict[str, CameraStream]  # pp24 camera key -> stream
    subtasks: list[tuple[int, str]]


@dataclasses.dataclass
class EpisodeResult:
    """What a worker hands back to the writer process."""

    source: EpisodeSource
    ok: bool
    reason: str | None = None
    prompt: str | None = None
    state: np.ndarray | None = None
    action: np.ndarray | None = None
    video_tmp: dict[str, str] = dataclasses.field(default_factory=dict)
    image_stats: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    qa: dict[str, Any] = dataclasses.field(default_factory=dict)


# --------------------------------------------------------------------------------------
# MCAP reading
# --------------------------------------------------------------------------------------


def _camera_topics(topics: set[str], top_eye: str) -> tuple[str, dict[str, str]]:
    """Return (station, {pp24 camera key: ABC topic})."""
    if "/top-camera" in topics:
        station, top = "realsense", "/top-camera"
    elif f"/top-{top_eye}-camera" in topics:
        station, top = "zedx", f"/top-{top_eye}-camera"
    else:
        raise ValueError(f"no top camera topic among {sorted(t for t in topics if 'camera' in t)}")
    return station, {CAM_TOP: top, CAM_LEFT_WRIST: "/left-wrist-camera", CAM_RIGHT_WRIST: "/right-wrist-camera"}


def read_episode(src: EpisodeSource, top_eye: str) -> RawEpisode:
    """Read robot streams, compressed camera packets, metadata and annotations of one episode."""
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    with open(src.mcap, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        summary = reader.get_summary()
        topics = {ch.topic for ch in summary.channels.values()} if summary else set()
        missing = [t for t in ROBOT_TOPICS if t not in topics]
        if missing:
            raise ValueError(f"missing robot topics {missing}")
        station, cam_topics = _camera_topics(topics, top_eye)
        for key, topic in cam_topics.items():
            if topic not in topics:
                raise ValueError(f"missing camera topic {topic} for {key}")
        metadata: dict[str, str] = {}
        for rec in reader.iter_metadata():
            metadata.update(dict(rec.metadata))

        topic_to_key = {v: k for k, v in cam_topics.items()}
        info_topics = {f"{t}-info": k for t, k in topic_to_key.items()}
        wanted = set(ROBOT_TOPICS) | set(cam_topics.values()) | set(info_topics) | {"/instruction"}
        robot_ts: dict[str, list[int]] = {t: [] for t in ROBOT_TOPICS}
        robot_val: dict[str, list[list[float]]] = {t: [] for t in ROBOT_TOPICS}
        widths = dict(STATE_TOPICS + ACTION_TOPICS)
        cams = {k: CameraStream(topic=t, codec="", timestamps=np.empty(0, np.int64), packets=[]) for k, t in cam_topics.items()}
        cam_ts: dict[str, list[int]] = {k: [] for k in cam_topics}
        instruction: str | None = None
        for _schema, channel, message, dec in reader.iter_decoded_messages(topics=sorted(wanted)):
            topic = channel.topic
            if topic in robot_ts:
                pos = list(dec.position)
                n = widths[topic]
                if len(pos) < n:
                    raise ValueError(f"{topic} position has {len(pos)} values, expected {n}")
                robot_ts[topic].append(message.log_time)
                robot_val[topic].append(pos[:n])
            elif topic in topic_to_key:
                key = topic_to_key[topic]
                fmt = _CODEC_FOR_FORMAT.get(str(dec.format).lower())
                if fmt is None:
                    raise ValueError(f"{topic}: unsupported video format {dec.format!r}")
                stream = cams[key]
                if not stream.codec:
                    stream.codec = fmt
                elif stream.codec != fmt:
                    raise ValueError(f"{topic}: codec changes mid-stream ({stream.codec} -> {fmt})")
                cam_ts[key].append(message.log_time)
                stream.packets.append(bytes(dec.data))
            elif topic in info_topics:
                cams[info_topics[topic]].calib = {
                    "width": int(dec.width),
                    "height": int(dec.height),
                    "distortion_model": str(dec.distortion_model),
                    "K": [float(x) for x in dec.K],
                    "D": [float(x) for x in dec.D],
                    "frame_id": str(dec.frame_id),
                }
            elif topic == "/instruction":
                instruction = str(dec.data)

    robot: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for topic in ROBOT_TOPICS:
        ts = np.asarray(robot_ts[topic], dtype=np.int64)
        val = np.asarray(robot_val[topic], dtype=np.float64).reshape(len(ts), widths[topic])
        if len(ts) == 0:
            raise ValueError(f"{topic} has no messages")
        order = np.argsort(ts, kind="stable")
        ts, val = ts[order], val[order]
        keep = np.ones(len(ts), dtype=bool)  # drop duplicate stamps, keep the last sample
        keep[:-1] = ts[1:] != ts[:-1]
        robot[topic] = (ts[keep], val[keep])
    for key, stream in cams.items():
        stream.timestamps = np.asarray(cam_ts[key], dtype=np.int64)
        if len(stream.packets) == 0:
            raise ValueError(f"{stream.topic} has no frames")
        if np.any(np.diff(stream.timestamps) < 0):
            raise ValueError(f"{stream.topic}: timestamps are not in decode order")

    subtasks: list[tuple[int, str]] = []
    if src.annotation is not None:
        with open(src.annotation, "rb") as f:
            areader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _s, ch, msg, dec in areader.iter_decoded_messages(topics=["/subtask-annotation"]):
                subtasks.append((int(msg.log_time), str(dec.data)))
    return RawEpisode(metadata=metadata, instruction=instruction, station=station, robot=robot, cameras=cams, subtasks=subtasks)


# --------------------------------------------------------------------------------------
# Time alignment
# --------------------------------------------------------------------------------------


def build_time_grid(stream_ts: Iterable[np.ndarray], fps: int, max_frames: int | None) -> np.ndarray:
    """Uniform grid (int64 ns) covering the interval where every stream has data."""
    stream_ts = list(stream_ts)
    t0 = max(int(ts[0]) for ts in stream_ts)
    t1 = min(int(ts[-1]) for ts in stream_ts)
    if t1 < t0:
        return np.empty(0, dtype=np.int64)
    period = 1e9 / fps
    n = int(math.floor((t1 - t0) / period + 1e-9)) + 1
    if max_frames is not None:
        n = min(n, max_frames)
    return t0 + np.round(np.arange(n) * period).astype(np.int64)


def nearest_indices(src_ts: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    """Index of the nearest source timestamp for every query (ties -> earlier sample)."""
    if len(src_ts) == 1:
        return np.zeros(len(query_ts), dtype=np.int64)
    idx = np.searchsorted(src_ts, query_ts, side="left")
    idx = np.clip(idx, 1, len(src_ts) - 1)
    left, right = src_ts[idx - 1], src_ts[idx]
    take_left = (query_ts - left) <= (right - query_ts)
    return np.where(take_left, idx - 1, idx).astype(np.int64)


def sample_stream(ts: np.ndarray, values: np.ndarray, grid: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Resample one robot stream onto the grid. Returns (values [N, D], |dt| to nearest sample in ms)."""
    idx = nearest_indices(ts, grid)
    err_ms = np.abs(ts[idx] - grid) / 1e6
    if mode == "nearest":
        return values[idx], err_ms
    if mode == "linear":
        rel = (ts - grid[0]) / 1e9
        q = (grid - grid[0]) / 1e9
        out = np.stack([np.interp(q, rel, values[:, d]) for d in range(values.shape[1])], axis=1)
        return out, err_ms
    raise ValueError(f"unknown align mode {mode!r}")


def _err_summary(err_ms: np.ndarray) -> dict[str, float]:
    return {"mean_ms": round(float(err_ms.mean()), 3), "max_ms": round(float(err_ms.max()), 3)}


# --------------------------------------------------------------------------------------
# LeRobot v2.1 episode statistics (mirrors lerobot.common.datasets.compute_stats)
# --------------------------------------------------------------------------------------


def estimate_num_samples(n: int, min_num_samples: int = 100, max_num_samples: int = 10_000, power: float = 0.75) -> int:
    if n < min_num_samples:
        min_num_samples = n
    return max(min_num_samples, min(int(n**power), max_num_samples))


def sample_indices(n: int) -> list[int]:
    return np.round(np.linspace(0, n - 1, estimate_num_samples(n))).astype(int).tolist()


def auto_downsample_height_width(img: np.ndarray, target_size: int = 150, max_size_threshold: int = 300) -> np.ndarray:
    _, height, width = img.shape
    if max(width, height) < max_size_threshold:
        return img
    factor = int(width / target_size) if width > height else int(height / target_size)
    return img[:, ::factor, ::factor]


def feature_stats(array: np.ndarray, axis: int | tuple[int, ...], keepdims: bool) -> dict[str, np.ndarray]:
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims),
        "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims),
        "std": np.std(array, axis=axis, keepdims=keepdims),
        "count": np.array([len(array)]),
    }


def image_stats(frames_chw: np.ndarray) -> dict[str, np.ndarray]:
    st = feature_stats(frames_chw, axis=(0, 2, 3), keepdims=True)
    return {k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in st.items()}


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# --------------------------------------------------------------------------------------
# Video: decode -> crop/resize -> AV1 encode
# --------------------------------------------------------------------------------------

_PIL_FILTERS = {"area": "BOX", "bilinear": "BILINEAR", "bicubic": "BICUBIC", "lanczos": "LANCZOS"}


def resize_rgb(rgb: np.ndarray, height: int, width: int, aspect: str, resize_filter: str) -> np.ndarray:
    """Crop/pad/stretch an HxWx3 uint8 frame to the target aspect ratio and resize it."""
    from PIL import Image

    resample = getattr(Image.Resampling, _PIL_FILTERS[resize_filter])
    h, w = rgb.shape[:2]
    if (h, w) == (height, width):
        return rgb
    target = width / height
    if aspect == "crop" and abs(w / h - target) > 1e-6:
        if w / h > target:  # too wide: centre-crop columns
            new_w = int(round(h * target))
            x0 = (w - new_w) // 2
            rgb = rgb[:, x0 : x0 + new_w]
        else:  # too tall: centre-crop rows
            new_h = int(round(w / target))
            y0 = (h - new_h) // 2
            rgb = rgb[y0 : y0 + new_h]
    elif aspect == "pad" and abs(w / h - target) > 1e-6:
        scale = min(width / w, height / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        small = np.asarray(Image.fromarray(rgb).resize((nw, nh), resample))
        out = np.zeros((height, width, 3), dtype=np.uint8)
        y0, x0 = (height - nh) // 2, (width - nw) // 2
        out[y0 : y0 + nh, x0 : x0 + nw] = small
        return out
    return np.asarray(Image.fromarray(np.ascontiguousarray(rgb)).resize((width, height), resample))


def transcode_camera(
    stream: CameraStream,
    frame_idx: np.ndarray,
    out_path: Path,
    cfg: ConvertConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Decode `stream`, emit source frame `frame_idx[k]` as output frame k, encode to `out_path`.

    `frame_idx` must be non-decreasing (it is: nearest-neighbour lookups of a monotonic grid).
    Returns LeRobot image stats (of the frames fed to the encoder) and QA counters.
    """
    import av
    from av.video.reformatter import VideoReformatter

    reformatter = VideoReformatter()  # reused: ~4x faster than VideoFrame.to_ndarray(format=...)
    n_out = len(frame_idx)
    sample_set = sample_indices(n_out)
    sample_pos: dict[int, list[int]] = {}
    for pos, k in enumerate(sample_set):
        sample_pos.setdefault(k, []).append(pos)
    samples = np.empty((len(sample_set), 3, cfg.height, cfg.width), dtype=np.uint8)

    dec = av.CodecContext.create(stream.codec, "r")
    dec.thread_type = "AUTO"
    if cfg.decoder_threads:
        dec.thread_count = cfg.decoder_threads

    out_path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(out_path), "w")
    options: dict[str, str] = {}
    if cfg.gop is not None:
        options["g"] = str(cfg.gop)
    if cfg.crf is not None:
        options["crf"] = str(cfg.crf)
    if cfg.svt_preset is not None and cfg.vcodec == "libsvtav1":
        options["preset"] = str(cfg.svt_preset)
    vstream = container.add_stream(cfg.vcodec, rate=cfg.fps, options=options)
    vstream.pix_fmt = cfg.pix_fmt
    vstream.width = cfg.width
    vstream.height = cfg.height
    time_base = Fraction(1, cfg.fps)

    k = 0  # next output frame
    held: np.ndarray | None = None  # last resized source frame (fallback for undecodable packets)
    qa = {"source_frames": len(stream.packets), "decode_errors": 0, "held_frames": 0, "decoded_frames": 0}
    src_size: tuple[int, int] | None = None

    def emit(img: np.ndarray, count_as_held: bool) -> None:
        nonlocal k
        vf = av.VideoFrame.from_ndarray(img, format="rgb24")
        vf.pts = k
        vf.time_base = time_base
        for packet in vstream.encode(vf):
            container.mux(packet)
        for pos in sample_pos.get(k, ()):
            samples[pos] = img.transpose(2, 0, 1)
        if count_as_held:
            qa["held_frames"] += 1
        k += 1

    def consume(frames: Iterable[Any]) -> None:
        nonlocal held, src_size
        for fr in frames:
            p = int(fr.pts) if fr.pts is not None else -1
            qa["decoded_frames"] += 1
            if k >= n_out or p < frame_idx[k]:
                continue  # not needed (e.g. 60 Hz source on a 30 Hz grid)
            img = None
            while k < n_out and frame_idx[k] <= p:
                if frame_idx[k] == p:
                    if img is None:
                        rgb = reformatter.reformat(fr, format="rgb24").to_ndarray()
                        src_size = (rgb.shape[0], rgb.shape[1])
                        img = resize_rgb(rgb, cfg.height, cfg.width, cfg.aspect, cfg.resize_filter)
                    emit(img, False)
                else:  # the needed packet produced no picture: hold the previous one
                    if held is None:
                        rgb = reformatter.reformat(fr, format="rgb24").to_ndarray()
                        src_size = (rgb.shape[0], rgb.shape[1])
                        held = resize_rgb(rgb, cfg.height, cfg.width, cfg.aspect, cfg.resize_filter)
                    emit(held, True)
            if img is not None:
                held = img

    last_needed = int(frame_idx[-1]) if n_out else -1
    for i, data in enumerate(stream.packets):
        if i > last_needed + 2 or k >= n_out:
            break
        pkt = av.Packet(data)
        pkt.pts = i
        pkt.dts = i
        try:
            consume(dec.decode(pkt))
        except av.error.FFmpegError as exc:  # corrupt access unit: skip it, hold previous frame
            qa["decode_errors"] += 1
            log.debug("%s packet %d: %s", stream.topic, i, exc)
    try:
        consume(dec.decode(None))
    except av.error.FFmpegError as exc:
        qa["decode_errors"] += 1
        log.debug("%s flush: %s", stream.topic, exc)
    if k < n_out:
        if held is None:
            container.close()
            raise RuntimeError(f"{stream.topic}: no decodable frame")
        while k < n_out:
            emit(held, True)
    for packet in vstream.encode():
        container.mux(packet)
    container.close()

    qa["source_size_hw"] = list(src_size) if src_size else None
    if len(frame_idx) > 1:
        qa["duplicated_frames"] = int(np.sum(np.diff(frame_idx) == 0))
    stats = image_stats(np.stack([auto_downsample_height_width(s) for s in samples]))
    return stats, qa


# --------------------------------------------------------------------------------------
# Episode conversion (runs in worker processes)
# --------------------------------------------------------------------------------------


def resolve_prompt(src: EpisodeSource, raw: RawEpisode, cfg: ConvertConfig) -> str:
    candidates: list[str | None]
    from_dir = src.task_dir.replace("_", " ")
    if cfg.prompt_source == "instruction":
        candidates = [raw.instruction, raw.metadata.get("instruction"), raw.metadata.get("task_name"), from_dir]
    elif cfg.prompt_source == "metadata":
        candidates = [raw.metadata.get("instruction"), raw.metadata.get("task_name"), raw.instruction, from_dir]
    else:
        candidates = [from_dir]
    for c in candidates:
        if c and c.strip():
            return " ".join(c.strip().split())
    return from_dir


def convert_episode(src: EpisodeSource, cfg: ConvertConfig, tmp_dir: str) -> EpisodeResult:
    """Fetch (Hub sources), read, align and transcode one episode.

    Never raises: failures are reported in the result. A successful result is also pickled to
    `<tmp_dir>/<episode_id>/result.pkl` so that `--resume` can skip it after an interruption.
    """
    import pickle

    try:
        t0 = time.time()
        src.fetch()
        fetch_s = time.time() - t0
        res = _convert_episode(src, cfg, tmp_dir)
    except Exception as exc:  # noqa: BLE001 - one bad episode must not kill the run
        import traceback

        shutil.rmtree(Path(tmp_dir) / src.episode_id, ignore_errors=True)
        return EpisodeResult(source=src, ok=False, reason=f"{type(exc).__name__}: {exc} | {traceback.format_exc(limit=3)}")
    if res.ok:
        res.qa["fetch_seconds"] = round(fetch_s, 2)
        res.qa["raw_bytes"] = sum(p.stat().st_size for p in (src.mcap, src.annotation) if p is not None and p.exists())
        with open(Path(tmp_dir) / src.episode_id / "result.pkl", "wb") as f:
            pickle.dump(res, f)
        if cfg.delete_raw:
            src.delete_download()
    return res


def load_cached_result(src: EpisodeSource, tmp_dir: str) -> EpisodeResult | None:
    """Result of an episode converted by an interrupted earlier run (see --resume)."""
    import pickle

    path = Path(tmp_dir) / src.episode_id / "result.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            res: EpisodeResult = pickle.load(f)
    except Exception:  # noqa: BLE001 - corrupt/partial pickle: convert again
        return None
    if not res.ok or not all(Path(v).exists() for v in res.video_tmp.values()):
        return None
    return dataclasses.replace(res, source=src)


def _convert_episode(src: EpisodeSource, cfg: ConvertConfig, tmp_dir: str) -> EpisodeResult:
    t_start = time.time()
    try:
        raw = read_episode(src, cfg.top_eye)
    except Exception as exc:  # noqa: BLE001 - report and skip broken episodes
        return EpisodeResult(source=src, ok=False, reason=f"read failed: {exc}")

    offset_ns = int(round(cfg.camera_offset_ms * 1e6))
    cam_ts = {key: stream.timestamps + offset_ns for key, stream in raw.cameras.items()}
    grid = build_time_grid([raw.robot[t][0] for t in ROBOT_TOPICS] + list(cam_ts.values()), cfg.fps, cfg.max_frames)
    n = len(grid)
    if n < cfg.min_frames:
        return EpisodeResult(source=src, ok=False, reason=f"only {n} aligned frames (< --min-frames {cfg.min_frames})")

    qa: dict[str, Any] = {
        "station": raw.station,
        "camera_topics": {k: s.topic for k, s in raw.cameras.items()},
        "camera_codecs": {k: s.codec for k, s in raw.cameras.items()},
        "calibration": {k: s.calib for k, s in raw.cameras.items()},
        "metadata": raw.metadata,
        "instruction": raw.instruction,
        "subtasks": [{"t": (t - int(grid[0])) / 1e9, "label": label} for t, label in raw.subtasks],
        "frames": n,
        "grid_start_unix_ns": int(grid[0]),
        "duration_s": round((int(grid[-1]) - int(grid[0])) / 1e9, 4),
        "source_rates_hz": {},
        "sync_error": {},
    }
    for topic in ROBOT_TOPICS:
        ts = raw.robot[topic][0]
        qa["source_rates_hz"][topic] = round(float((len(ts) - 1) / max((ts[-1] - ts[0]) / 1e9, 1e-9)), 2)
    for key, ts in cam_ts.items():
        qa["source_rates_hz"][raw.cameras[key].topic] = round(float((len(ts) - 1) / max((ts[-1] - ts[0]) / 1e9, 1e-9)), 2)

    def assemble(spec: Sequence[tuple[str, int]]) -> np.ndarray:
        parts = []
        for topic, _ in spec:
            vals, err = sample_stream(raw.robot[topic][0], raw.robot[topic][1], grid, cfg.align)
            qa["sync_error"][topic] = _err_summary(err)
            parts.append(vals)
        out = np.concatenate(parts, axis=1).astype(np.float32)
        if cfg.invert_gripper:
            out[:, GRIPPER_DIMS] = 1.0 - out[:, GRIPPER_DIMS]
        return out

    state = assemble(STATE_TOPICS)
    action = assemble(ACTION_TOPICS)
    if not (np.isfinite(state).all() and np.isfinite(action).all()):
        return EpisodeResult(source=src, ok=False, reason="non-finite state/action values")

    ep_tmp = Path(tmp_dir) / src.episode_id
    video_tmp: dict[str, str] = {}
    image_stats_out: dict[str, dict[str, Any]] = {}
    qa["video"] = {}
    try:
        for key in CAMERA_KEYS:
            stream = raw.cameras[key]
            idx = nearest_indices(cam_ts[key], grid)
            qa["sync_error"][stream.topic] = _err_summary(np.abs(cam_ts[key][idx] - grid) / 1e6)
            out_path = ep_tmp / f"{key}.mp4"
            stats, vqa = transcode_camera(stream, idx, out_path, cfg)
            video_tmp[key] = str(out_path)
            image_stats_out[key] = stats
            qa["video"][key] = vqa
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(ep_tmp, ignore_errors=True)
        return EpisodeResult(source=src, ok=False, reason=f"video failed: {exc}")

    qa["convert_seconds"] = round(time.time() - t_start, 2)
    return EpisodeResult(
        source=src,
        ok=True,
        prompt=resolve_prompt(src, raw, cfg),
        state=state,
        action=action,
        video_tmp=video_tmp,
        image_stats=image_stats_out,
        qa=qa,
    )


# --------------------------------------------------------------------------------------
# Dataset writer (main process)
# --------------------------------------------------------------------------------------


def probe_video_info(path: Path) -> dict[str, Any]:
    """`info` block of a video feature, exactly as lerobot v2.1 `get_video_info` builds it."""
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        pix_fmt = s.codec_context.pix_fmt
        channels = 1 if "gray" in pix_fmt else 4 if ("rgba" in pix_fmt or "yuva" in pix_fmt) else 3
        info: dict[str, Any] = {
            "video.height": int(s.codec_context.height),
            "video.width": int(s.codec_context.width),
            "video.codec": s.codec_context.codec.canonical_name,  # "av1" (not the decoder name, e.g. libdav1d)
            "video.pix_fmt": pix_fmt,
            "video.is_depth_map": False,
            "video.fps": int(s.base_rate),
            "video.channels": channels,
        }
        if c.streams.audio:
            raise ValueError(f"{path} unexpectedly has an audio stream")
        info["has_audio"] = False
        return info


def write_episode_parquet(
    path: Path,
    state: np.ndarray,
    action: np.ndarray,
    episode_index: int,
    global_offset: int,
    task_index: int,
    fps: int,
    hf_style: str,
) -> dict[str, np.ndarray]:
    """Write one episode parquet file; returns the column arrays (for statistics).

    hf_style 'sequence' reproduces `datasets<4` Dataset.to_parquet (what lerobot 0cf8648 + datasets<3,
    the training stack, writes): `huggingface` metadata with Sequence features, snappy, dictionary
    encoding, one row group per 1000 rows. 'list' reproduces `datasets>=4` (List features, page
    index, content-defined chunking) -- NOT readable by datasets<4. 'none' omits the metadata.
    """
    hf_meta = hf_features_metadata(hf_style)
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = len(state)
    frame_index = np.arange(n, dtype=np.int64)
    cols = {
        "observation.state": state.astype(np.float32),
        "action": action.astype(np.float32),
        "next.done": np.zeros(n, dtype=bool),
        "timestamp": (frame_index / fps).astype(np.float32),
        "frame_index": frame_index,
        "episode_index": np.full(n, episode_index, dtype=np.int64),
        "index": frame_index + global_offset,
        "task_index": np.full(n, task_index, dtype=np.int64),
    }
    cols["next.done"][-1] = True
    state_type = pa.list_(pa.field("element", pa.float32()), STATE_DIM)
    action_type = pa.list_(pa.field("element", pa.float32()), ACTION_DIM)
    arrays = [
        pa.FixedSizeListArray.from_arrays(pa.array(cols["observation.state"].reshape(-1), pa.float32()), type=state_type),
        pa.FixedSizeListArray.from_arrays(pa.array(cols["action"].reshape(-1), pa.float32()), type=action_type),
        pa.array(cols["next.done"], pa.bool_()),
        pa.array(cols["timestamp"], pa.float32()),
        pa.array(cols["frame_index"], pa.int64()),
        pa.array(cols["episode_index"], pa.int64()),
        pa.array(cols["index"], pa.int64()),
        pa.array(cols["task_index"], pa.int64()),
    ]
    schema = pa.schema(
        [
            pa.field("observation.state", state_type),
            pa.field("action", action_type),
            pa.field("next.done", pa.bool_()),
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
        ],
        metadata={"huggingface": hf_meta} if hf_meta else None,
    )
    table = pa.Table.from_arrays(arrays, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hf_style == "list":
        try:
            writer = pq.ParquetWriter(
                path,
                schema=schema,
                compression="snappy",
                use_dictionary=True,
                write_page_index=True,
                use_content_defined_chunking=DATASETS_CDC_OPTIONS,
            )
        except TypeError:  # pyarrow < 21 has no content-defined chunking
            writer = pq.ParquetWriter(path, schema=schema, compression="snappy", use_dictionary=True)
        else:
            writer.add_key_value_metadata({"content_defined_chunking": json.dumps(DATASETS_CDC_OPTIONS)})
        writer.write_table(table)
        writer.close()
    else:
        with pq.ParquetWriter(path, schema=schema, compression="snappy", use_dictionary=True) as writer:
            for start in range(0, n, DATASETS3_BATCH_SIZE):
                writer.write_table(table.slice(start, DATASETS3_BATCH_SIZE))
    cols["fps"] = np.float64(fps)
    return cols


def numeric_episode_stats(cols: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for key in ("observation.state", "action", "next.done"):
        arr = cols[key]
        out[key] = feature_stats(arr, axis=0, keepdims=arr.ndim == 1)
    out["timestamp"] = feature_stats(cols["frame_index"] / cols["fps"], axis=0, keepdims=True)
    for key in ("frame_index", "episode_index", "index", "task_index"):
        out[key] = feature_stats(cols[key], axis=0, keepdims=True)
    return out


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """One JSON object per line, as the `jsonlines` writer used by lerobot (ensure_ascii=False)."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_prompt_overrides(path: Path | None) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Optional --prompts-json: {"episodes": {uuid: prompt}, "tasks": {abc_task_dir: [paraphrase, ...]}}."""
    if path is None:
        return {}, {}
    data = json.loads(Path(path).read_text())
    episodes = {str(k): str(v) for k, v in data.get("episodes", {}).items()}
    tasks = {str(k): [str(p) for p in v] for k, v in data.get("tasks", {}).items()}
    return episodes, tasks


def pick_prompt(res: EpisodeResult, ep_prompts: dict[str, str], task_prompts: dict[str, list[str]]) -> str:
    eid = res.source.episode_id
    if eid in ep_prompts:
        return ep_prompts[eid]
    pool = task_prompts.get(res.source.task_dir) or task_prompts.get("*")
    if pool:  # deterministic per-episode paraphrase, mimicking pp24's per-episode prompts
        h = int(hashlib.sha1(eid.encode()).hexdigest(), 16)
        return pool[h % len(pool)]
    assert res.prompt is not None
    return res.prompt


def finalize_dataset(
    root: Path,
    results: list[EpisodeResult],
    cfg: ConvertConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Assign indices, move videos, write parquet + meta. Returns the summary dict."""
    ep_prompts, task_prompts = load_prompt_overrides(args.prompts_json)
    ok = [r for r in results if r.ok]
    if not ok:
        raise SystemExit("no episode converted successfully")

    tasks: dict[str, int] = {}
    episodes_rows, stats_rows, manifest_eps = [], [], []
    offset = 0
    video_info: dict[str, Any] | None = None
    for ep_idx, res in enumerate(ok):
        prompt = pick_prompt(res, ep_prompts, task_prompts)
        task_index = tasks.setdefault(prompt, len(tasks))
        chunk = ep_idx // args.chunks_size
        assert res.state is not None and res.action is not None
        cols = write_episode_parquet(
            root / DATA_PATH.format(episode_chunk=chunk, episode_index=ep_idx),
            res.state,
            res.action,
            ep_idx,
            offset,
            task_index,
            cfg.fps,
            args.hf_features_style,
        )
        for key in CAMERA_KEYS:
            dst = root / VIDEO_PATH.format(episode_chunk=chunk, video_key=key, episode_index=ep_idx)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(res.video_tmp[key], dst)
            if video_info is None:
                video_info = probe_video_info(dst)
        n = len(res.state)
        stats = numeric_episode_stats(cols)
        ordered = {}
        for key in pp24_features(cfg.height, cfg.width, {}):
            ordered[key] = res.image_stats[key] if key in CAMERA_KEYS else stats[key]
        stats_rows.append({"episode_index": ep_idx, "stats": _to_jsonable(ordered)})
        episodes_rows.append({"episode_index": ep_idx, "tasks": [prompt], "length": n})
        manifest_eps.append(
            {
                "episode_index": ep_idx,
                "source_episode": res.source.episode_id,
                "source_task": res.source.task_dir,
                "source_split": res.source.split,
                "source_mcap": str(res.source.mcap),
                "pp24_task": ".".join(ABC_TO_PP24[res.source.task_dir]) if res.source.task_dir in ABC_TO_PP24 else None,
                "prompt": prompt,
                **res.qa,
            }
        )
        offset += n

    assert video_info is not None
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    total_eps = len(ok)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": args.robot_type,
        "total_episodes": total_eps,
        "total_frames": offset,
        "total_tasks": len(tasks),
        "total_videos": total_eps * len(CAMERA_KEYS),
        "total_chunks": (total_eps - 1) // args.chunks_size + 1,
        "chunks_size": args.chunks_size,
        "fps": cfg.fps,
        "splits": {"train": f"0:{total_eps}"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": pp24_features(cfg.height, cfg.width, video_info),
    }
    with open(meta / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)
    write_jsonl(meta / "tasks.jsonl", ({"task_index": i, "task": t} for t, i in tasks.items()))
    write_jsonl(meta / "episodes.jsonl", episodes_rows)
    write_jsonl(meta / "episodes_stats.jsonl", stats_rows)

    summary = {
        "converter": "abc130k_to_lerobot.py",
        "created_unix": int(time.time()),
        "config": dataclasses.asdict(cfg),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "src"},
        "src": [str(s) for s in args.src],
        "converted": total_eps,
        "skipped": [{"source_episode": r.source.episode_id, "source_mcap": str(r.source.mcap), "reason": r.reason} for r in results if not r.ok],
        "total_frames": offset,
        "episodes": manifest_eps,
    }
    with open(root / MANIFEST_NAME, "w") as f:
        json.dump(_to_jsonable(summary), f, indent=1)
    return summary


# --------------------------------------------------------------------------------------
# Source discovery
# --------------------------------------------------------------------------------------


def discover_local(path: Path) -> list[EpisodeSource]:
    if path.is_file() and path.name.endswith(".mcap"):
        return [EpisodeSource.from_mcap(path)]
    if (path / "episode.mcap").exists():
        return [EpisodeSource.from_mcap(path / "episode.mcap")]
    return [EpisodeSource.from_mcap(p) for p in sorted(path.rglob("episode.mcap"))]


def discover_hf(task: str, split: str, download_dir: Path, max_episodes: int | None, select: str, seed: int, only: set[str] | None) -> list[EpisodeSource]:
    """List `data/<split>/<task>/episode_*/*.mcap` on the Hub (one recursive listing).

    Nothing is downloaded here: every worker fetches its own episode right before converting
    it (EpisodeSource.fetch), so downloads overlap with transcoding and, with --delete-raw,
    disk use stays bounded by the number of workers.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    base = f"data/{split}/{task}"
    files: dict[str, list[tuple[str, int]]] = {}
    for e in api.list_repo_tree(ABC_REPO_ID, repo_type="dataset", path_in_repo=base, recursive=True):
        if e.path.endswith(".mcap") and hasattr(e, "size"):
            files.setdefault(e.path.rsplit("/", 1)[0], []).append((e.path, int(e.size or 0)))
    eps = sorted(files)
    if only:
        eps = [e for e in eps if any(e.rsplit("/", 1)[-1].removeprefix("episode_").startswith(o) for o in only)]
    if max_episodes is not None and max_episodes < len(eps):
        if select == "random":
            rng = np.random.default_rng(seed)
            eps = sorted(rng.choice(eps, size=max_episodes, replace=False).tolist())
        else:
            eps = eps[:max_episodes]
    out = []
    for ep in eps:
        names = {p.rsplit("/", 1)[-1] for p, _ in files[ep]}
        ep_dir = download_dir / ep
        out.append(
            EpisodeSource(
                mcap=ep_dir / "episode.mcap",
                annotation=ep_dir / "annotation.mcap" if "annotation.mcap" in names else None,
                task_dir=task,
                split=split,
                episode_id=ep.rsplit("/", 1)[-1].removeprefix("episode_"),
                hub_files=tuple(sorted(p for p, _ in files[ep])),
                download_dir=str(download_dir),
                hub_bytes=sum(sz for _, sz in files[ep]),
            )
        )
    log.info("%s/%s: %d episodes selected (%.1f GB of MCAP on the Hub)", split, task, len(out), sum(e.hub_bytes for e in out) / 1e9)
    return out


def resolve_sources(args: argparse.Namespace) -> list[EpisodeSource]:
    only = set(args.episodes) if args.episodes else None
    sources: list[EpisodeSource] = []
    for s in args.src:
        p = Path(os.path.expanduser(s))
        if not s.startswith("hf:") and p.exists():
            found = discover_local(p)
            if only:
                found = [e for e in found if any(e.episode_id.startswith(o) for o in only)]
            found = sorted(found, key=lambda e: (e.task_dir, e.episode_id))
            if args.max_episodes is not None:
                found = found[: args.max_episodes]
            sources.extend(found)
        else:
            task = s.removeprefix("hf:")
            split = args.split
            if "/" in task:
                split, task = task.split("/", 1)
            sources.extend(discover_hf(task, split, Path(os.path.expanduser(args.download_dir)), args.max_episodes, args.select, args.seed, only))
    # de-duplicate, keep order
    seen: set[Path] = set()
    unique = []
    for e in sources:
        if e.mcap.resolve() not in seen:
            seen.add(e.mcap.resolve())
            unique.append(e)
    return unique


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_image_size(s: str) -> tuple[int, int]:
    h, w = s.lower().split("x")
    return int(h), int(w)


def default_repo_id(sources: list[EpisodeSource]) -> str:
    """`abc130k/<abc_task_dir>` (release naming), or `abc130k/mixed` for several tasks."""
    tasks = {e.task_dir for e in sources}
    return f"abc130k/{tasks.pop()}" if len(tasks) == 1 else "abc130k/mixed"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("input")
    g.add_argument("--src", nargs="+", required=True, help="local episode/task/root dirs (searched for */episode.mcap) or Hub task names ('hf:<task>' or 'hf:<split>/<task>')")
    g.add_argument("--split", default="train", help="Hub split for bare task names")
    g.add_argument("--download-dir", default="~/.cache/abc130k_raw", help="where Hub episodes are downloaded (keeps the repo layout)")
    g.add_argument("--max-episodes", type=int, default=None, help="max episodes per --src entry")
    g.add_argument("--select", choices=["first", "random"], default="first", help="which Hub episodes to take with --max-episodes")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--episodes", nargs="*", default=None, help="only these episode UUIDs (prefixes ok)")

    g = p.add_argument_group("output")
    g.add_argument("--out", default=os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot"), help="root directory; the dataset is written to <out>/<repo-id> (default: $HF_LEROBOT_HOME)")
    g.add_argument("--repo-id", default=None, help="dataset id; default abc130k/<abc_task_dir> (abc130k/mixed for several tasks)")
    g.add_argument("--robot-type", default="xdof")
    g.add_argument("--chunks-size", type=int, default=DEFAULT_CHUNKS_SIZE)
    g.add_argument("--hf-features-style", choices=["sequence", "list", "none"], default="sequence", help="parquet `huggingface` schema metadata/writer: 'sequence' = datasets<4 (the training stack: lerobot 0cf8648 + datasets<3), 'list' = datasets>=4 (unreadable by datasets<4), 'none' = omit")
    g.add_argument("--overwrite", action="store_true", help="delete an existing output dataset first")
    g.add_argument("--resume", action="store_true", help="reuse episodes already converted by an interrupted run (<out>/<repo-id>/.tmp)")
    g.add_argument("--delete-raw", action="store_true", help="delete each Hub-downloaded episode once converted (local --src data is never deleted)")
    g.add_argument("--dry-run", action="store_true", help="list the selected episodes and download size, then exit")

    g = p.add_argument_group("time alignment")
    g.add_argument("--fps", type=int, default=30)
    g.add_argument("--align", choices=["nearest", "linear"], default="nearest", help="robot-stream resampling onto the fps grid (cameras: always nearest frame)")
    g.add_argument("--camera-offset-ms", type=float, default=0.0, help="added to camera timestamps before matching (e.g. to model capture latency)")
    g.add_argument("--max-frames", type=int, default=None, help="truncate episodes to this many frames")
    g.add_argument("--min-frames", type=int, default=2, help="skip episodes with fewer aligned frames")

    g = p.add_argument_group("images")
    g.add_argument("--image-size", type=parse_image_size, default=(168, 224), help="HxW of every camera")
    g.add_argument("--aspect", choices=["crop", "stretch", "pad"], default="crop", help="how to reach HxW aspect from other aspects (ZED-X 16:10); 4:3 RealSense frames are only resized")
    g.add_argument("--resize-filter", choices=sorted(_PIL_FILTERS), default="area")
    g.add_argument("--top-eye", choices=["left", "right"], default="left", help="ZED-X stereo top camera eye used as left_image")
    g.add_argument("--vcodec", default="libsvtav1")
    g.add_argument("--pix-fmt", default="yuv420p")
    g.add_argument("--crf", type=int, default=30)
    g.add_argument("--gop", type=int, default=2)
    g.add_argument("--svt-preset", type=int, default=None, help="SVT-AV1 preset (None = encoder default, as lerobot)")
    g.add_argument("--decoder-threads", type=int, default=4, help="frame threads per decoder (0 = FFmpeg auto)")

    g = p.add_argument_group("state / action / prompt")
    g.add_argument("--invert-gripper", action="store_true", help="store 1-aperture (0=open). pp24 and ABC both use 0=closed, 1=open")
    g.add_argument("--prompt-source", choices=["instruction", "metadata", "task-dir"], default="instruction")
    g.add_argument("--prompts-json", type=Path, default=None, help='optional {"episodes": {uuid: prompt}, "tasks": {abc_task: [paraphrases]}}')

    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "filelock", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    os.environ.setdefault("SVT_LOG", "1")  # SVT-AV1: errors only
    try:
        import av

        av.logging.set_level(av.logging.ERROR)
    except Exception:  # noqa: BLE001
        pass

    height, width = args.image_size
    cfg = ConvertConfig(
        fps=args.fps,
        height=height,
        width=width,
        aspect=args.aspect,
        resize_filter=args.resize_filter,
        top_eye=args.top_eye,
        align=args.align,
        camera_offset_ms=args.camera_offset_ms,
        invert_gripper=args.invert_gripper,
        max_frames=args.max_frames,
        min_frames=args.min_frames,
        vcodec=args.vcodec,
        pix_fmt=args.pix_fmt,
        crf=args.crf,
        gop=args.gop,
        svt_preset=args.svt_preset,
        decoder_threads=args.decoder_threads,
        prompt_source=args.prompt_source,
        delete_raw=args.delete_raw,
    )
    sources = resolve_sources(args)
    if not sources:
        log.error("no episodes found for --src %s", args.src)
        return 1
    repo_id = args.repo_id or default_repo_id(sources)
    root = Path(os.path.expanduser(args.out)) / repo_id
    hub_bytes = sum(e.hub_bytes for e in sources)
    if args.dry_run:
        log.info("dry run: %d episodes (%d from the Hub, %.1f GB to download) -> %s", len(sources), sum(bool(e.hub_files) for e in sources), hub_bytes / 1e9, root)
        return 0
    tmp = root / ".tmp"
    if root.exists():
        if args.resume and tmp.is_dir() and not (root / "meta" / "info.json").exists():
            log.info("resuming into %s", root)
        elif args.overwrite:
            shutil.rmtree(root)
        else:
            log.error("%s exists (use --overwrite, or --resume after an interrupted run)", root)
            return 1
    tmp.mkdir(parents=True, exist_ok=True)

    results: list[EpisodeResult | None] = [None] * len(sources)
    todo = []
    for i, src in enumerate(sources):
        cached = load_cached_result(src, str(tmp)) if args.resume else None
        if cached is not None:
            results[i] = cached
        else:
            todo.append(i)
    log.info("converting %d episodes (%d cached) -> %s (%d workers)", len(todo), len(sources) - len(todo), root, args.workers)
    t0 = time.time()
    if args.workers <= 1:
        for done, i in enumerate(todo, 1):
            results[i] = convert_episode(sources[i], cfg, str(tmp))
            _log_result(results[i], done, len(todo))
    else:
        import multiprocessing as mp

        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
            futs = {pool.submit(convert_episode, sources[i], cfg, str(tmp)): i for i in todo}
            for done, fut in enumerate(as_completed(futs), 1):
                i = futs[fut]
                results[i] = fut.result()
                _log_result(results[i], done, len(todo))
    final = [r for r in results if r is not None]
    summary = finalize_dataset(root, final, cfg, args)
    shutil.rmtree(tmp, ignore_errors=True)
    log.info(
        "done in %.1fs: %d episodes, %d frames, %d skipped -> %s",
        time.time() - t0,
        summary["converted"],
        summary["total_frames"],
        len(summary["skipped"]),
        root,
    )
    return 0


def _log_result(res: EpisodeResult | None, done: int, total: int) -> None:
    if res is None:
        return
    if res.ok:
        q = res.qa
        cam_err = max(v["max_ms"] for k, v in q["sync_error"].items() if "camera" in k)
        log.info(
            "[%d/%d] %s %s: %d frames (%s, cam sync max %.1f ms, fetch %.1fs, convert %.1fs)",
            done, total, res.source.task_dir, res.source.episode_id[:8], q["frames"], q["station"], cam_err, q.get("fetch_seconds", 0.0), q["convert_seconds"],
        )
    else:
        log.warning("[%d/%d] %s %s skipped: %s", done, total, res.source.task_dir, res.source.episode_id[:8], res.reason)


if __name__ == "__main__":
    sys.exit(main())
