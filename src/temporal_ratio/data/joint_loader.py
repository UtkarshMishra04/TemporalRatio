# Adapted from src/temporal_ratio/data/action_loader.py
#
# Self-contained data loader for LeRobot datasets (LIBERO, RoboCasa, etc.)
# NO imports from openpi or cosmos_predict2.

from __future__ import annotations

import dataclasses
import json
import logging
import multiprocessing
import os
import pathlib
import typing

from temporal_ratio._project_root import PROJECT_ROOT
from collections.abc import Iterator, Sequence
from typing import Any, Literal

import einops
import numpy as np
import pydantic
import torch
import torch.utils.data

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalization stats (replaces temporal_ratio.data.normalize)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class NormStats:
    """Per-key normalization statistics."""
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None


class _PydanticNormStats(pydantic.BaseModel):
    mean: list
    std: list
    q01: list | None = None
    q99: list | None = None

    def to_norm_stats(self) -> NormStats:
        return NormStats(
            mean=np.asarray(self.mean, dtype=np.float32),
            std=np.asarray(self.std, dtype=np.float32),
            q01=np.asarray(self.q01, dtype=np.float32) if self.q01 is not None else None,
            q99=np.asarray(self.q99, dtype=np.float32) if self.q99 is not None else None,
        )


class _NormStatsDict(pydantic.BaseModel):
    norm_stats: dict[str, _PydanticNormStats]


def load_norm_stats(directory: str | pathlib.Path) -> dict[str, NormStats]:
    """Load norm_stats.json from *directory* and return ``{key: NormStats}``."""
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    raw = _NormStatsDict(**json.loads(path.read_text()))
    return {k: v.to_norm_stats() for k, v in raw.norm_stats.items()}


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Zero-pad along *axis* so ``x.shape[axis] == target_dim``."""
    if x.shape[axis] >= target_dim:
        return x
    pad_width = [(0, 0)] * len(x.shape)
    pad_width[axis] = (0, target_dim - x.shape[axis])
    return np.pad(x, pad_width, constant_values=value)


def normalize_quantile(x: np.ndarray, stats: NormStats) -> np.ndarray:
    """Quantile normalization to [-1, 1]."""
    assert stats.q01 is not None and stats.q99 is not None
    q01 = stats.q01[..., : x.shape[-1]]
    q99 = stats.q99[..., : x.shape[-1]]
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def normalize_zscore(x: np.ndarray, stats: NormStats) -> np.ndarray:
    mean = stats.mean[..., : x.shape[-1]]
    std = stats.std[..., : x.shape[-1]]
    return (x - mean) / (std + 1e-6)


# ---------------------------------------------------------------------------
# Dataset configuration (replaces temporal_ratio.config.registry.DataConfig subset)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DataConfig:
    """Minimal, self-contained data configuration."""

    # LeRobot repo_id (e.g. "physical-intelligence/libero").
    repo_id: str = ""

    # Dataset type determines how images are repacked.
    # Supported: "libero", "robocasa"
    dataset_type: str = "libero"

    # Horizons
    action_horizon: int = 17
    observation_horizon: int = 17

    # Keys inside the LeRobot dataset used to build delta_timestamps.
    action_sequence_keys: tuple[str, ...] = ("actions",)
    obs_sequence_keys: tuple[str, ...] = ("image", "wrist_image")

    # If True, extract prompt from LeRobot task_index.
    prompt_from_task: bool = True

    # Frame skip for temporal subsampling (1 = consecutive frames).
    frame_skip: int = 1

    # If True, concatenate views horizontally (flat); otherwise tile in a 2x2 grid.
    flat_frames: bool = True

    # Pad progress bar into the bottom-right quadrant (only when flat_frames=False).
    pad_progress: bool = False

    # Path to a directory containing ``norm_stats.json``.
    norm_stats_dir: str | None = None

    # Use quantile normalization instead of z-score.
    use_quantile_norm: bool = True

    # Model action dimension (for zero-padding states/actions).
    model_action_dim: int = 256

    # Per-camera resize (height, width).  Applied to each camera view before stacking.
    resize_height: int = 224
    resize_width: int = 224

    # Extra repo_ids for multi-task ConcatDataset.
    extra_repo_ids: tuple[str, ...] = ()

    # Number of DataLoader workers.
    num_workers: int = 4

    # Batch size (global, before DDP splitting).
    batch_size: int = 16

    # Random seed.
    seed: int = 42


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _parse_image(image: np.ndarray) -> np.ndarray:
    """Ensure image is uint8, (... H W C)."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 4:
        if image.shape[1] == 3:
            image = einops.rearrange(image, "b c h w -> b h w c")
    elif image.ndim == 3:
        if image.shape[0] == 3:
            image = einops.rearrange(image, "c h w -> h w c")
    return image


def _resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize image(s) to (height, width) using PIL, preserving aspect ratio with padding.

    Accepts shapes (..., H, W, C).  Returns same leading dims with new H, W.
    """
    from PIL import Image

    leading = image.shape[:-3]
    flat = image.reshape(-1, *image.shape[-3:])
    results = []
    for img in flat:
        pil = Image.fromarray(img)
        # Resize keeping aspect ratio, pad remainder with black.
        scale = min(height / pil.height, width / pil.width)
        new_h, new_w = int(pil.height * scale), int(pil.width * scale)
        pil = pil.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        canvas.paste(pil, ((width - new_w) // 2, (height - new_h) // 2))
        results.append(np.asarray(canvas))
    out = np.stack(results, axis=0).reshape(*leading, height, width, image.shape[-1])
    return out


# ---------------------------------------------------------------------------
# Prompt extraction from LeRobot task_index
# ---------------------------------------------------------------------------

class PromptFromTask:
    """Callable that injects ``prompt`` into a sample dict using ``task_index``."""

    def __init__(self, tasks: dict[int, str]):
        self.tasks = tasks

    def __call__(self, data: dict) -> dict:
        if "task_index" not in data:
            return data
        task_index = int(data["task_index"])
        prompt = self.tasks.get(task_index)
        if prompt is None:
            raise ValueError(f"task_index={task_index} not in task mapping")
        data["prompt"] = prompt
        return data


# ---------------------------------------------------------------------------
# Per-sample transform: repack keys, parse images, combine views
# ---------------------------------------------------------------------------

class LiberoRepackTransform:
    """Repack a raw LeRobot LIBERO sample into the training dict format.

    Output keys: state, image/{base_0_rgb, left_wrist_0_rgb}, actions, prompt,
                 frame_index, episode_index.
    """

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        out: dict[str, Any] = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
            },
        }
        if "actions" in data:
            out["actions"] = data["actions"]
        if "prompt" in data:
            out["prompt"] = data["prompt"]
        if "frame_index" in data:
            out["frame_index"] = data["frame_index"]
        if "episode_index" in data:
            out["episode_index"] = data["episode_index"]
        return out


class RobocasaRepackTransform:
    """Repack a raw LeRobot RoboCasa sample."""

    def __call__(self, data: dict) -> dict:
        primary = _parse_image(data["observation/primary_image"])
        secondary = _parse_image(data["observation/secondary_image"])
        wrist = _parse_image(data["observation/wrist_image"])
        out: dict[str, Any] = {
            "state": data["observation/state"],
            "image": {
                "left_base_0_rgb": primary,
                "right_base_0_rgb": secondary,
                "left_wrist_0_rgb": wrist,
            },
        }
        if "actions" in data:
            out["actions"] = data["actions"]
        if "prompt" in data:
            out["prompt"] = data["prompt"]
        if "frame_index" in data:
            out["frame_index"] = data["frame_index"]
        if "episode_index" in data:
            out["episode_index"] = data["episode_index"]
        return out


# ---------------------------------------------------------------------------
# Per-sample transform: combine camera views into a single video tensor,
#                        resize, normalize state/actions, pad dims.
# ---------------------------------------------------------------------------

class CombineViewsTransform:
    """Combines multi-view images into a single tensor and optionally normalizes state/actions.

    After this transform the sample contains:
        - ``wan_images_latents``: (T, H_out, W_out, C) uint8  (or H_out = H, W_out = W*nviews for flat)
        - ``state``: (state_dim,) float32, possibly normalized + padded
        - ``actions``: (action_horizon, action_dim) float32, possibly normalized + padded
        - ``prompt``: str
        - ``wan_tokenized_prompt``: None  (placeholder; actual tokenization happens in the model)
    """

    def __init__(
        self,
        dataset_type: str = "libero",
        flat_frames: bool = True,
        pad_progress: bool = False,
        norm_stats: dict[str, NormStats] | None = None,
        use_quantile_norm: bool = True,
        model_action_dim: int = 256,
        resize_height: int = 224,
        resize_width: int = 224,
        dataset: Any = None,
    ):
        self.dataset_type = dataset_type
        self.flat_frames = flat_frames
        self.pad_progress = pad_progress
        self.norm_stats = norm_stats
        self.use_quantile_norm = use_quantile_norm
        self.model_action_dim = model_action_dim
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.dataset = dataset  # needed for pad_progress episode length lookup

    # ---- helpers ----
    def _normalize_key(self, key: str, arr: np.ndarray) -> np.ndarray:
        if self.norm_stats is None or key not in self.norm_stats:
            return arr
        stats = self.norm_stats[key]
        if self.use_quantile_norm:
            return normalize_quantile(arr, stats)
        return normalize_zscore(arr, stats)

    def _resize_views(self, views: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Resize each camera view to (resize_height, resize_width)."""
        return {k: _resize_with_pad(v, self.resize_height, self.resize_width) for k, v in views.items()}

    # ---- main ----
    def __call__(self, data: dict) -> dict:
        views = self._resize_views(data["image"])

        if self.dataset_type == "robocasa":
            combined = self._combine_robocasa(views, data)
        else:
            combined = self._combine_libero(views, data)

        # Normalize state and actions
        state = np.asarray(data["state"], dtype=np.float32)
        state = self._normalize_key("state", state)
        state = pad_to_dim(state, self.model_action_dim, axis=-1)

        out: dict[str, Any] = {
            "wan_images_latents": combined,
            "wan_tokenized_prompt": None,
            "state": state,
        }

        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            actions = self._normalize_key("actions", actions)
            actions = pad_to_dim(actions, self.model_action_dim, axis=-1)
            out["actions"] = actions

        if "prompt" in data:
            out["prompt"] = data["prompt"]

        if "frame_index" in data:
            out["frame_index"] = data["frame_index"]
        if "episode_index" in data:
            out["episode_index"] = data["episode_index"]

        return out

    def _combine_libero(self, views: dict[str, np.ndarray], data: dict) -> np.ndarray:
        base = views["base_0_rgb"]       # (T, H, W, C)
        wrist = views["left_wrist_0_rgb"]  # (T, H, W, C)
        T, H, W, C = base.shape

        if self.flat_frames:
            combined = np.concatenate([base, wrist], axis=2)  # (T, H, W*2, C)
        else:
            combined = np.zeros((T, H * 2, W * 2, C), dtype=base.dtype)
            combined[:, :H, :W, :] = base
            combined[:, H:, :W, :] = wrist

        if self.pad_progress and not self.flat_frames:
            combined = self._add_progress_bar(combined, data, H, W, T, C)

        return combined

    def _combine_robocasa(self, views: dict[str, np.ndarray], data: dict) -> np.ndarray:
        left_base = views["left_base_0_rgb"]
        right_base = views["right_base_0_rgb"]
        wrist = views["left_wrist_0_rgb"]
        T, H, W, C = left_base.shape

        if self.flat_frames:
            combined = np.concatenate([left_base, right_base, wrist], axis=2)
        else:
            combined = np.zeros((T, H * 2, W * 2, C), dtype=left_base.dtype)
            combined[:, :H, :W, :] = left_base
            combined[:, :H, W:, :] = right_base
            combined[:, H:, :W, :] = wrist

        if self.pad_progress and not self.flat_frames:
            combined = self._add_progress_bar(combined, data, H, W, T, C)

        return combined

    def _add_progress_bar(
        self, combined: np.ndarray, data: dict, H: int, W: int, T: int, C: int
    ) -> np.ndarray:
        """Render a vertical progress bar in the bottom-right quadrant."""
        if self.dataset is None:
            return combined
        episode_idx = data.get("episode_index")
        start_index = data.get("frame_index")
        if episode_idx is None or start_index is None:
            return combined
        end_index = start_index + data["actions"].shape[0] - 1

        # Try to get episode length
        try:
            if hasattr(self.dataset, "get_episode_info"):
                episode_length = self.dataset.get_episode_info(episode_idx)
            elif hasattr(self.dataset, "meta"):
                episode_length = self.dataset.meta.episodes[episode_idx]
            else:
                return combined
        except Exception:
            return combined

        # Update prompt with progress info
        if "prompt" in data:
            data["prompt"] = (
                data["prompt"]
                + f" Current progress is from {start_index} to {end_index} out of {episode_length} frames."
            )

        bar = np.ones((T, H, W, C), dtype=combined.dtype) * 255
        start_h = max(0, min(H, int(float(start_index / episode_length) * H)))
        end_h = max(0, min(H, int(float(end_index / episode_length) * H)))
        if end_h > start_h:
            bar[:, start_h:end_h, :, :] = 0
        combined[:, -H:, -W:, :] = bar
        return combined


# ---------------------------------------------------------------------------
# Full per-sample pipeline: repack -> combine views -> normalize/pad
# ---------------------------------------------------------------------------

class SampleTransform:
    """End-to-end per-sample transform wrapping repack + combine."""

    def __init__(self, config: DataConfig, norm_stats: dict[str, NormStats] | None = None, dataset: Any = None):
        if config.dataset_type == "robocasa":
            self.repack = RobocasaRepackTransform()
        else:
            self.repack = LiberoRepackTransform()

        self.combine = CombineViewsTransform(
            dataset_type=config.dataset_type,
            flat_frames=config.flat_frames,
            pad_progress=config.pad_progress,
            norm_stats=norm_stats,
            use_quantile_norm=config.use_quantile_norm,
            model_action_dim=config.model_action_dim,
            resize_height=config.resize_height,
            resize_width=config.resize_width,
            dataset=dataset,
        )

    def __call__(self, data: dict) -> dict:
        data = self.repack(data)
        data = self.combine(data)
        return data


# ---------------------------------------------------------------------------
# Transformed dataset wrappers
# ---------------------------------------------------------------------------

class TransformedLeRobotDataset(torch.utils.data.Dataset):
    """Wraps a LeRobot dataset with prompt extraction + sample transform."""

    def __init__(
        self,
        dataset: lerobot_dataset.LeRobotDataset,
        prompt_transform: PromptFromTask | None,
        sample_transform: SampleTransform,
    ):
        self.dataset = dataset
        self.prompt_transform = prompt_transform
        self.sample_transform = sample_transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        item = self.dataset[index]
        if self.prompt_transform is not None:
            item = self.prompt_transform(item)
        return self.sample_transform(item)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def _collate_fn(items: list[dict]) -> dict:
    """Stack list-of-dicts into a batched dict of numpy arrays.

    Strings (prompts) are collected into a list. ``None`` values are passed through.
    """
    keys = items[0].keys()
    batch: dict[str, Any] = {}
    for k in keys:
        vals = [item[k] for item in items]
        if vals[0] is None:
            batch[k] = None
        elif isinstance(vals[0], str):
            batch[k] = vals
        elif isinstance(vals[0], np.ndarray):
            batch[k] = np.stack(vals, axis=0)
        elif isinstance(vals[0], (int, float, np.integer, np.floating)):
            batch[k] = np.array(vals)
        else:
            # Fallback: try stacking, else keep as list
            try:
                batch[k] = np.stack([np.asarray(v) for v in vals], axis=0)
            except Exception:
                batch[k] = vals
    return batch


def _worker_init_fn(worker_id: int) -> None:
    """Prevent JAX GPU preallocation inside DataLoader workers (if JAX is imported)."""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    rank = os.environ.get("RANK", "0")
    logger.debug(f"[Rank {rank}] DataLoader worker {worker_id} started.")


# ---------------------------------------------------------------------------
# Observation container (simple dataclass replacing openpi model.Observation)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Observation:
    """Minimal observation container returned by the data loader.

    All tensor fields are numpy arrays with a leading batch dimension.
    """
    state: np.ndarray                      # (B, state_dim)
    wan_images_latents: np.ndarray         # (B, T, H, W, C) uint8
    wan_tokenized_prompt: Any = None       # None placeholder

    def to_torch(self, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> "Observation":
        """Convert numpy arrays to torch tensors."""
        return Observation(
            state=torch.as_tensor(self.state, dtype=dtype, device=device),
            wan_images_latents=torch.as_tensor(self.wan_images_latents, device=device),
            wan_tokenized_prompt=self.wan_tokenized_prompt,
        )


# ---------------------------------------------------------------------------
# Public API: create_data_loader
# ---------------------------------------------------------------------------

def create_data_loader(
    config: DataConfig,
    *,
    shuffle: bool = True,
    num_batches: int | None = None,
) -> Iterator[tuple[Observation, np.ndarray, list[str]]]:
    """Create an iterable data loader that yields ``(observation, actions, prompts)``.

    Args:
        config: Data configuration.
        shuffle: Whether to shuffle the data.
        num_batches: If set, stop after this many batches.  Otherwise loop forever.

    Yields:
        (Observation, actions, prompts) where:
            - Observation has .state (B, D), .wan_images_latents (B, T, H, W, C)
            - actions is np.ndarray (B, action_horizon, action_dim)
            - prompts is list[str] of length B
    """
    # 1. Load normalization stats ------------------------------------------
    norm_stats: dict[str, NormStats] | None = None
    if config.norm_stats_dir is not None:
        try:
            norm_stats = load_norm_stats(config.norm_stats_dir)
            logger.info(f"Loaded norm stats from {config.norm_stats_dir}")
        except FileNotFoundError:
            logger.warning(f"Norm stats not found at {config.norm_stats_dir}, skipping normalization.")

    # 2. Build LeRobot dataset ---------------------------------------------
    meta = lerobot_dataset.LeRobotDatasetMetadata(config.repo_id)

    delta_timestamps: dict[str, list[float]] = {}
    for key in config.action_sequence_keys:
        delta_timestamps[key] = [t / meta.fps for t in range(config.action_horizon)]
    for key in config.obs_sequence_keys:
        delta_timestamps[key] = [t * config.frame_skip / meta.fps for t in range(config.observation_horizon)]

    dataset = lerobot_dataset.LeRobotDataset(config.repo_id, delta_timestamps=delta_timestamps)

    # Prompt extraction
    prompt_transform: PromptFromTask | None = None
    if config.prompt_from_task:
        prompt_transform = PromptFromTask(meta.tasks)

    # Per-sample transform
    sample_transform = SampleTransform(config, norm_stats=norm_stats, dataset=dataset)
    full_dataset: torch.utils.data.Dataset = TransformedLeRobotDataset(dataset, prompt_transform, sample_transform)

    # Extra repo_ids (multi-task) ------------------------------------------
    if config.extra_repo_ids:
        all_datasets: list[torch.utils.data.Dataset] = [full_dataset]
        for rid in config.extra_repo_ids:
            extra_meta = lerobot_dataset.LeRobotDatasetMetadata(rid)
            extra_ds = lerobot_dataset.LeRobotDataset(rid, delta_timestamps=delta_timestamps)
            extra_prompt = PromptFromTask(extra_meta.tasks) if config.prompt_from_task else None
            extra_transform = SampleTransform(config, norm_stats=norm_stats, dataset=extra_ds)
            all_datasets.append(TransformedLeRobotDataset(extra_ds, extra_prompt, extra_transform))
            logger.info(f"  Added extra dataset {rid} ({len(extra_ds)} frames)")
        full_dataset = torch.utils.data.ConcatDataset(all_datasets)

    # 3. Build DataLoader --------------------------------------------------
    sampler = None
    local_batch_size = config.batch_size

    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        sampler = torch.utils.data.distributed.DistributedSampler(
            full_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=True,
        )
        # Note: local_batch_size stays the same (user should set per-GPU batch size).

    mp_context = None
    if config.num_workers > 0:
        mp_context = multiprocessing.get_context("spawn")

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    loader = torch.utils.data.DataLoader(
        typing.cast(torch.utils.data.Dataset, full_dataset),
        batch_size=local_batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=config.num_workers,
        multiprocessing_context=mp_context,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=4 if config.num_workers > 0 else None,
        collate_fn=_collate_fn,
        worker_init_fn=_worker_init_fn,
        drop_last=True,
        generator=generator,
    )

    # 4. Yield batches -----------------------------------------------------
    num_yielded = 0
    while True:
        for batch in loader:
            if num_batches is not None and num_yielded >= num_batches:
                return

            obs = Observation(
                state=batch["state"],
                wan_images_latents=batch["wan_images_latents"],
                wan_tokenized_prompt=batch.get("wan_tokenized_prompt"),
            )
            actions = batch.get("actions")
            prompts = batch.get("prompt", [""] * local_batch_size)
            if isinstance(prompts, np.ndarray):
                prompts = prompts.tolist()

            num_yielded += 1
            yield obs, actions, prompts

        # Update sampler epoch for reshuffling in DDP
        if sampler is not None:
            sampler.set_epoch(sampler.epoch + 1 if hasattr(sampler, "epoch") else 0)


# ---------------------------------------------------------------------------
# Convenience: quick test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    cfg = DataConfig(
        repo_id="physical-intelligence/libero",
        dataset_type="libero",
        action_horizon=17,
        observation_horizon=17,
        obs_sequence_keys=("image", "wrist_image"),
        action_sequence_keys=("actions",),
        prompt_from_task=True,
        flat_frames=True,
        norm_stats_dir=str(PROJECT_ROOT / "assets" / "pi05_libero" / "physical-intelligence" / "libero"),
        batch_size=4,
        num_workers=0,
    )

    loader = create_data_loader(cfg, shuffle=True, num_batches=3)
    for i, (obs, actions, prompts) in enumerate(loader):
        print(f"Batch {i}:")
        print(f"  state:              {obs.state.shape}  dtype={obs.state.dtype}")
        print(f"  wan_images_latents: {obs.wan_images_latents.shape}  dtype={obs.wan_images_latents.dtype}")
        print(f"  actions:            {actions.shape}  dtype={actions.dtype}")
        print(f"  prompts[0]:         {prompts[0][:80]}...")
        print()
