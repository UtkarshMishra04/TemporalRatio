"""Self-contained video dataset loading for joint training.

Supports:
  - LeRobot datasets (multi-camera, flat layout)
  - Local video datasets (MP4 files + text prompts)
  - ConcatDataset for mixing multiple sources
  - WeightedDistributedSampler for weighted sampling across datasets
  - Standard and weighted DataLoader creation

No dependency on cosmos_predict2 or imaginaire.

Ported from: third_party/cosmos-predict2.5/cosmos_predict2/experiments/base/cosmos_lerobot_assets_flat.py
"""

import os
import logging
import math
import traceback
from typing import Any, Callable, Iterator, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import ConcatDataset, DataLoader, Dataset, DistributedSampler, Sampler

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class ToTensorVideo:
    """Convert uint8 (T,C,H,W) → float32 (T,C,H,W) / 255."""
    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        if clip.dtype == torch.uint8:
            return clip.float() / 255.0
        return clip


class ResizePreprocess:
    """Resize video frames to target (H, W)."""
    def __init__(self, size: tuple[int, int]):
        self.size = size

    def __call__(self, video_frames: torch.Tensor) -> torch.Tensor:
        if video_frames.ndim == 4:
            return torch.stack([
                F.interpolate(f.unsqueeze(0), size=self.size, mode="bilinear", align_corners=False, antialias=True).squeeze(0)
                for f in video_frames
            ])
        return F.interpolate(video_frames.unsqueeze(0), size=self.size, mode="bilinear", align_corners=False, antialias=True).squeeze(0)


# ---------------------------------------------------------------------------
# Multi-Repo Video Dataset (Fast local-cache loader)
# ---------------------------------------------------------------------------

class MultiRepoVideoDataset(Dataset):
    """Fast multi-repo dataset that reads local HF cache directly.

    Scans meta/episodes.jsonl + meta/tasks.jsonl (instant), loads video frames
    with decord at __getitem__ time. Avoids per-repo LeRobotDataset init.

    Returns same dict format as LeRobotVideoDatasetFlat:
        - video: uint8 tensor [C, T, H_out, W_out]
        - ai_caption: str
        - fps: float
        - image_size: tensor [h, w, h, w]
        - num_frames: int
        - padding_mask: zeros tensor [1, h, w]
    """

    #: LeRobot datasets root (``$HF_LEROBOT_HOME``, default ``~/.cache/huggingface/lerobot``).
    CACHE_ROOT = os.environ.get("HF_LEROBOT_HOME", os.path.expanduser("~/.cache/huggingface/lerobot"))

    def __init__(
        self,
        repo_ids: List[str],
        num_frames: int = 17,
        video_size: tuple[int, int] = (224, 448),
        layout: str = "horizontal",
        frame_skip: int = 1,
        cache_root: Optional[str] = None,
    ):
        import json as _json

        super().__init__()
        self.num_frames = num_frames
        self.frame_skip = frame_skip
        self.layout = layout
        self.video_size = video_size
        self.num_failed_loads = 0

        cache_root = cache_root or self.CACHE_ROOT
        span = (num_frames - 1) * frame_skip + 1

        self.episodes = []  # (repo_dir, ep_idx, length, task_str, camera_keys)
        self.repo_episode_ranges = []  # (start, end) per repo
        self.fps = 30.0

        import os
        for repo_id in repo_ids:
            repo_dir = os.path.join(cache_root, repo_id)
            meta_dir = os.path.join(repo_dir, "meta")

            try:
                with open(os.path.join(meta_dir, "info.json")) as f:
                    info = _json.load(f)
                self.fps = info.get("fps", 30)
                chunks_size = info.get("chunks_size", 1000)

                # Discover camera keys from video directory
                vid_chunk0 = os.path.join(repo_dir, "videos", "chunk-000")
                camera_keys = sorted([
                    d for d in os.listdir(vid_chunk0)
                    if os.path.isdir(os.path.join(vid_chunk0, d))
                ])

                # Read task description (prefer non-empty task)
                with open(os.path.join(meta_dir, "tasks.jsonl")) as f:
                    tasks = {}
                    for line in f:
                        t = _json.loads(line)
                        tasks[t["task_index"]] = t["task"]
                # Use first non-empty task, or tasks from episode
                task_str = ""
                for idx in sorted(tasks.keys()):
                    if tasks[idx]:
                        task_str = tasks[idx]
                        break

                # Read episodes
                start = len(self.episodes)
                with open(os.path.join(meta_dir, "episodes.jsonl")) as f:
                    for line in f:
                        ep = _json.loads(line)
                        ep_len = ep["length"]
                        if ep_len >= span:
                            ep_idx = ep["episode_index"]
                            # Per-episode task override if available
                            ep_task = task_str
                            if ep.get("tasks"):
                                ep_task = ep["tasks"][0] if ep["tasks"][0] else task_str
                            chunk_idx = ep_idx // chunks_size
                            self.episodes.append((
                                repo_dir, ep_idx, ep_len, ep_task,
                                camera_keys, chunk_idx,
                            ))
                end = len(self.episodes)
                self.repo_episode_ranges.append((start, end))

            except FileNotFoundError:
                log.warning(f"  SKIP {repo_id} (not in local cache)")
                self.repo_episode_ranges.append((len(self.episodes), len(self.episodes)))

        self.preprocess = T.Compose([ToTensorVideo(), ResizePreprocess(video_size)])
        log.info(
            f"MultiRepoVideoDataset: {len(self.episodes)} valid episodes "
            f"from {len(repo_ids)} repos"
        )

    def __len__(self) -> int:
        return len(self.episodes)

    def __str__(self) -> str:
        return f"MultiRepoVideoDataset({len(self.episodes)} episodes, {len(self.repo_episode_ranges)} repos)"

    def _getitem_inner(self, index: int) -> dict:
        from decord import VideoReader, cpu
        import os

        repo_dir, ep_idx, ep_len, task_str, camera_keys, chunk_idx = self.episodes[index]
        span = (self.num_frames - 1) * self.frame_skip + 1
        local_start = np.random.randint(0, ep_len - span + 1)
        frame_ids = list(range(local_start, local_start + span, self.frame_skip))

        # Load each camera view with decord
        camera_frames = []
        for key in camera_keys:
            vid_path = os.path.join(
                repo_dir, "videos", f"chunk-{chunk_idx:03d}", key,
                f"episode_{ep_idx:06d}.mp4",
            )
            vr = VideoReader(vid_path, ctx=cpu(0), num_threads=1)
            frames = vr.get_batch(frame_ids).asnumpy()  # (T, H, W, C) uint8
            camera_frames.append(
                torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T, C, H, W)
            )

        # Combine cameras
        if self.layout == "horizontal":
            combined = torch.cat(camera_frames, dim=-1)
        else:
            combined = camera_frames[0]

        # Resize: uint8 → float/255 → resize → *255 → uint8
        combined = self.preprocess(combined)
        combined = torch.clamp(combined * 255.0, 0, 255).to(torch.uint8)

        video = combined.permute(1, 0, 2, 3)  # (C, T, H, W)
        _, _, h, w = video.shape

        return {
            "video": video,
            "ai_caption": task_str,
            "fps": float(self.fps),
            "image_size": torch.tensor([h, w, h, w]),
            "num_frames": self.num_frames,
            "padding_mask": torch.zeros(1, h, w),
        }

    def __getitem__(self, index: int) -> dict:
        for attempt in range(10):
            try:
                return self._getitem_inner(index)
            except Exception as e:
                self.num_failed_loads += 1
                log.warning(f"MultiRepo load failed idx={index} attempt={attempt}: {e}")
                index = np.random.randint(len(self.episodes))
        raise RuntimeError(f"Failed after 10 retries (total: {self.num_failed_loads})")


# ---------------------------------------------------------------------------
# LeRobot Video Dataset (Flat Layout)
# ---------------------------------------------------------------------------

class LeRobotVideoDatasetFlat(Dataset):
    """Dataset that loads LeRobot episodes and returns flattened multi-camera video.

    Each sample is a random sub-sequence of num_frames from one episode.
    Camera views are concatenated into a flat layout.

    Returns dict with:
        - video: uint8 tensor [C, T, H_out, W_out]
        - ai_caption: str (task description)
        - fps: float
        - image_size: tensor [h, w, h, w]
        - num_frames: int
        - padding_mask: zeros tensor [1, h, w]
    """

    def __init__(
        self,
        repo_id: str,
        num_frames: int = 17,
        video_size: tuple[int, int] = (224, 448),
        camera_keys: Optional[List[str]] = None,
        layout: str = "horizontal",
        frame_skip: int = 1,
        video_backend: Optional[str] = None,
        augment: bool = False,
        root: Optional[str] = None,
        wrist_blackout_prob: float = 0.0,
        cross_task_prob: float = 0.0,
    ):
        super().__init__()
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        from temporal_ratio.data import video_decode

        video_decode.install()

        self.repo_id = repo_id
        self.sequence_length = num_frames
        self.frame_skip = frame_skip
        self.layout = layout
        self.video_size = video_size
        self.wrist_blackout_prob = wrist_blackout_prob
        self.cross_task_prob = cross_task_prob

        lerobot_kwargs = {"repo_id": repo_id}
        if video_backend is not None:
            lerobot_kwargs["video_backend"] = video_backend
        if root is not None:
            lerobot_kwargs["root"] = root
        log.info(f"Loading LeRobot dataset: {repo_id}")
        self.lerobot_dataset = LeRobotDataset(**lerobot_kwargs)

        self.camera_keys = camera_keys or list(self.lerobot_dataset.meta.camera_keys)
        if not self.camera_keys:
            raise ValueError(f"No camera keys in {repo_id}")

        self.fps = self.lerobot_dataset.meta.fps
        self.num_episodes = self.lerobot_dataset.num_episodes

        # Build episode index
        self.episodes = []
        for ep_idx in range(self.num_episodes):
            start = self.lerobot_dataset.episode_data_index["from"][ep_idx].item()
            end = self.lerobot_dataset.episode_data_index["to"][ep_idx].item()
            self.episodes.append((start, end, end - start))

        # Filter valid episodes
        span = (self.sequence_length - 1) * self.frame_skip + 1
        self.valid_episodes = [
            (i, s, e, l) for i, (s, e, l) in enumerate(self.episodes) if l >= span
        ]
        if not self.valid_episodes:
            raise ValueError(f"No episodes >= {span} frames in {repo_id}")

        self.num_failed_loads = 0
        self._all_tasks = []
        if cross_task_prob > 0:
            seen = set()
            for ep_idx in range(min(self.num_episodes, 5000)):
                task = self.lerobot_dataset[self.episodes[ep_idx][0]].get("task", "")
                if task and task not in seen:
                    seen.add(task)
                    self._all_tasks.append(task)

        self.preprocess = T.Compose([ToTensorVideo(), ResizePreprocess(video_size)])
        self.augment_transform = T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.05) if augment else None
        self.augment_crop_scale = (0.9, 0.9) if augment else None

        log.info(f"  {repo_id}: {len(self.valid_episodes)} valid episodes, cameras={self.camera_keys}")

    def __len__(self) -> int:
        return len(self.valid_episodes)

    def __str__(self) -> str:
        return f"LeRobotVideoDatasetFlat({self.repo_id}, {len(self)} eps)"

    def _combine_cameras(self, stacked: dict[str, torch.Tensor]) -> torch.Tensor:
        views = [stacked[k] for k in self.camera_keys]
        if self.layout == "horizontal":
            return torch.cat(views, dim=-1)
        elif self.layout == "quad":
            T_dim, C, H, W = views[0].shape
            hh, hw = H // 2, W // 2
            crops = [v[:, :, :hh, :hw] for v in views[:3]]
            while len(crops) < 3:
                crops.append(torch.zeros(T_dim, C, hh, hw, dtype=views[0].dtype))
            return torch.cat(crops, dim=-1)
        raise ValueError(f"Unknown layout: {self.layout}")

    def _getitem_inner(self, index: int) -> dict:
        _, ep_start, ep_end, ep_len = self.valid_episodes[index]
        span = (self.sequence_length - 1) * self.frame_skip + 1
        local_start = np.random.randint(0, ep_len - span + 1)
        global_indices = list(range(ep_start + local_start, ep_start + local_start + span, self.frame_skip))

        camera_frames = {key: [] for key in self.camera_keys}
        caption = None
        for gidx in global_indices:
            item = self.lerobot_dataset[gidx]
            if caption is None:
                caption = item.get("task", "")
            for key in self.camera_keys:
                camera_frames[key].append(item[key])

        stacked = {k: torch.stack(v) for k, v in camera_frames.items()}

        # Wrist blackout
        if self.wrist_blackout_prob > 0 and np.random.random() < self.wrist_blackout_prob:
            wrist_keys = [k for k in self.camera_keys if "wrist" in k.lower()]
            for wk in wrist_keys:
                stacked[wk] = torch.zeros_like(stacked[wk])

        combined = self._combine_cameras(stacked)
        combined = torch.clamp(combined * 255.0, 0, 255).to(torch.uint8)

        if self.cross_task_prob > 0 and self._all_tasks and np.random.random() < self.cross_task_prob:
            caption = self._all_tasks[np.random.randint(len(self._all_tasks))]

        frames = self.preprocess(combined)
        if self.augment_crop_scale is not None:
            _, _, fh, fw = frames.shape
            scale = np.random.uniform(*self.augment_crop_scale)
            ch, cw = int(fh * scale), int(fw * scale)
            top, left = np.random.randint(0, fh - ch + 1), np.random.randint(0, fw - cw + 1)
            frames = torch.nn.functional.interpolate(
                frames[:, :, top:top+ch, left:left+cw], size=self.video_size, mode="bilinear", align_corners=False
            )
        if self.augment_transform is not None:
            frames = torch.stack([self.augment_transform(f) for f in frames])
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)

        video = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
        _, _, h, w = video.shape
        return {
            "video": video,
            "ai_caption": caption or "",
            "fps": self.fps,
            "image_size": torch.tensor([h, w, h, w]),
            "num_frames": self.sequence_length,
            "padding_mask": torch.zeros(1, h, w),
        }

    def __getitem__(self, index: int) -> dict:
        for attempt in range(10):
            try:
                return self._getitem_inner(index)
            except Exception as e:
                self.num_failed_loads += 1
                log.warning(f"[{self.repo_id}] Load failed idx={index} attempt={attempt}: {e}")
                index = np.random.randint(len(self.valid_episodes))
        raise RuntimeError(f"Failed after 10 retries (total: {self.num_failed_loads})")


# ---------------------------------------------------------------------------
# Local Video Dataset (MP4 files + text prompts)
# ---------------------------------------------------------------------------

class LocalVideoDatasetFlat(Dataset):
    """Dataset loading local MP4 videos + text prompt files.

    Expects directory structure:
        dataset_dir/
            videos/  *.mp4
            metas/   *.txt  (one prompt per video, matched by sorted order)

    Returns same dict format as LeRobotVideoDatasetFlat.
    """

    def __init__(
        self,
        dataset_dir: str,
        num_frames: int = 17,
        video_size: tuple[int, int] = (224, 448),
        frame_skip: int = 1,
    ):
        super().__init__()
        import os
        self.dataset_dir = dataset_dir
        self.sequence_length = num_frames
        self.frame_skip = frame_skip
        self.video_size = video_size

        video_dir = os.path.join(dataset_dir, "videos")
        meta_dir = os.path.join(dataset_dir, "metas")
        self.video_paths = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")])
        self.meta_paths = sorted([os.path.join(meta_dir, f) for f in os.listdir(meta_dir) if f.endswith(".txt")])
        self.num_failed_loads = 0

        self.preprocess = T.Compose([ToTensorVideo(), ResizePreprocess(video_size)])
        log.info(f"LocalVideoDatasetFlat: {len(self.video_paths)} videos from {dataset_dir}")

    def __len__(self) -> int:
        return len(self.video_paths)

    def __str__(self) -> str:
        return f"LocalVideoDatasetFlat({self.dataset_dir}, {len(self)} videos)"

    def __getitem__(self, index: int) -> dict:
        from decord import VideoReader, cpu
        import os

        for attempt in range(5):
            try:
                vr = VideoReader(self.video_paths[index], ctx=cpu(0), num_threads=2)
                total = len(vr)
                span = (self.sequence_length - 1) * self.frame_skip + 1
                if total < span:
                    index = np.random.randint(len(self))
                    continue

                start = np.random.randint(0, total - span + 1)
                ids = list(range(start, start + span, self.frame_skip))
                frames = torch.from_numpy(vr.get_batch(ids).asnumpy())  # (T, H, W, C) uint8
                frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)
                frames = self.preprocess(frames)
                frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
                video = frames.permute(1, 0, 2, 3)  # (C, T, H, W)

                # Load caption
                caption = ""
                if index < len(self.meta_paths):
                    caption = open(self.meta_paths[index]).read().strip()

                _, _, h, w = video.shape
                return {
                    "video": video,
                    "ai_caption": caption,
                    "fps": float(vr.get_avg_fps()),
                    "image_size": torch.tensor([h, w, h, w]),
                    "num_frames": self.sequence_length,
                    "padding_mask": torch.zeros(1, h, w),
                }
            except Exception as e:
                self.num_failed_loads += 1
                log.warning(f"Load failed {self.video_paths[index]}: {e}")
                index = np.random.randint(len(self))

        raise RuntimeError(f"Failed after 5 retries")


# ---------------------------------------------------------------------------
# WeightedDistributedSampler
# ---------------------------------------------------------------------------

class WeightedDistributedSampler(Sampler):
    """Distributed sampler with per-dataset weights for ConcatDataset."""

    def __init__(
        self,
        datasets: List[Dataset],
        weights: List[float],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        num_samples: Optional[int] = None,
    ):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0

        self.datasets = datasets
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.weights /= self.weights.sum()
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        self.dataset_lengths = [len(d) for d in datasets]
        self.offsets = [0]
        for length in self.dataset_lengths[:-1]:
            self.offsets.append(self.offsets[-1] + length)
        self.total_size = sum(self.dataset_lengths)
        self.num_samples = num_samples or self.total_size
        self.num_samples_per_replica = int(math.ceil(self.num_samples / self.num_replicas))
        self.total_num_samples = self.num_samples_per_replica * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        dataset_indices = torch.multinomial(self.weights, self.total_num_samples, replacement=True, generator=g)
        dataset_counts = torch.bincount(dataset_indices, minlength=len(self.datasets))

        all_indices = []
        for i, count in enumerate(dataset_counts):
            if count > 0:
                local = torch.randint(0, self.dataset_lengths[i], (int(count.item()),), generator=g)
                all_indices.append(local + self.offsets[i])

        if all_indices:
            all_indices = torch.cat(all_indices)
            perm = torch.randperm(len(all_indices), generator=g)
            all_indices = all_indices[perm]
            indices = all_indices[self.rank:self.total_num_samples:self.num_replicas].tolist()
        else:
            indices = []
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples_per_replica

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


# ---------------------------------------------------------------------------
# DataLoader creation helpers
# ---------------------------------------------------------------------------

def get_sampler(dataset: Dataset, seed: int = 0) -> DistributedSampler:
    """Create a standard DistributedSampler."""
    num_replicas = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    return DistributedSampler(dataset, num_replicas=num_replicas, rank=rank, shuffle=True, seed=seed)


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    sampler: Optional[Sampler] = None,
    seed: int = 0,
    **kwargs,
) -> DataLoader:
    """Create a standard DataLoader with distributed sampler (shuffled with ``seed``)."""
    if sampler is None:
        sampler = get_sampler(dataset, seed)
    return DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False,
        sampler=sampler, num_workers=num_workers, pin_memory=pin_memory,
        drop_last=drop_last, **kwargs,
    )


def create_weighted_dataloader(
    datasets: List[Dataset],
    weights: List[float],
    batch_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    seed: int = 0,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader with weighted sampling across multiple datasets."""
    concat = ConcatDataset(datasets)
    sampler = WeightedDistributedSampler(datasets=datasets, weights=weights, seed=seed)
    return DataLoader(
        dataset=concat, batch_size=batch_size, shuffle=False,
        sampler=sampler, num_workers=num_workers, pin_memory=pin_memory,
        drop_last=drop_last, **kwargs,
    )
