# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LeRobot dataset loader for Cosmos Predict2 video training.

Loads episodes directly from a LeRobot dataset (HuggingFace repo_id),
extracts camera frames, concatenates them into a flat multi-view layout,
and returns the same dict format expected by the Cosmos training loop.

Supports:
  - Any number of cameras (auto-detected from dataset.meta.camera_keys)
  - Configurable flat layout: 2-camera (horizontal concat) or 3-camera (half-quad)
  - Frame skip for wider temporal coverage
  - Task descriptions as captions
"""

import os
import traceback
from typing import Any, Callable, Optional, List

import numpy as np
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms as T

from cosmos_video._src.imaginaire.lazy_config import LazyCall as L
from cosmos_video._src.imaginaire.utils import log
from cosmos_video._src.predict2.datasets.local_datasets.dataset_utils import ResizePreprocess, ToTensorVideo

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    LeRobotDataset = None


class LeRobotVideoDatasetFlat(Dataset):
    """Dataset that loads LeRobot episodes and returns flattened multi-camera video.

    Each sample is one episode (or a random sub-sequence of num_frames from it).
    Camera views are concatenated into a flat layout matching the existing
    VideoDatasetFlat format.

    Layout modes:
      - "horizontal": Concatenate all cameras horizontally → (H, W*N_cams, 3)
      - "quad": 2x2 grid → top-left, top-right, bottom-left cameras + blank bottom-right
        Then flatten to 3 horizontal strips of (H/2, W/2) each → (H/2, W/2 * 3)

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
        num_frames: int,
        video_size: tuple[int, int],
        camera_keys: Optional[List[str]] = None,
        layout: str = "quad",  # "horizontal" or "quad"
        frame_skip: int = 1,
        video_backend: Optional[str] = None,
        wrist_blackout_prob: float = 0.0,
        wrist_camera_keys: Optional[List[str]] = None,
        augment: bool = False,
    ) -> None:
        """
        Args:
            repo_id: HuggingFace repo ID for the LeRobot dataset.
            num_frames: Number of frames to sample per episode.
            video_size: Target (H, W) after resize.
            camera_keys: Which camera keys to use. None = auto-detect all.
            layout: How to arrange multiple cameras ("horizontal" or "quad").
            frame_skip: Sample every Nth frame (1 = consecutive).
            video_backend: Video decoder backend ("torchcodec" or "pyav").
                None = auto-detect. Use "pyav" for av1-encoded videos.
            wrist_blackout_prob: Probability of blacking out wrist camera views
                per sample (0.0 = never, 1.0 = always).
            wrist_camera_keys: Which camera keys are wrist cameras. None = auto-detect
                by looking for "wrist" in key name.
            augment: If True, apply color jitter and Gaussian blur augmentation.
        """
        super().__init__()

        if LeRobotDataset is None:
            raise ImportError(
                "lerobot is not installed. Install with: pip install lerobot"
            )

        self.repo_id = repo_id
        self.sequence_length = num_frames
        self.frame_skip = frame_skip
        self.layout = layout

        # Load the LeRobot dataset
        log.info(f"Loading LeRobot dataset: {repo_id}")
        lerobot_kwargs = {"repo_id": repo_id}
        if video_backend is not None:
            lerobot_kwargs["video_backend"] = video_backend
        self.lerobot_dataset = LeRobotDataset(**lerobot_kwargs)

        # Determine camera keys
        if camera_keys is not None:
            self.camera_keys = camera_keys
        else:
            self.camera_keys = list(self.lerobot_dataset.meta.camera_keys)

        if not self.camera_keys:
            raise ValueError(
                f"No camera keys found in dataset {repo_id}. "
                f"Available keys: {list(self.lerobot_dataset.meta.camera_keys)}"
            )

        # Wrist blackout augmentation
        self.wrist_blackout_prob = wrist_blackout_prob
        if wrist_camera_keys is not None:
            self.wrist_camera_keys = wrist_camera_keys
        else:
            self.wrist_camera_keys = [k for k in self.camera_keys if "wrist" in k.lower()]

        self.fps = self.lerobot_dataset.meta.fps
        self.num_episodes = self.lerobot_dataset.num_episodes

        # Build episode index: list of (start_frame, end_frame, episode_length)
        self.episodes: list[tuple[int, int, int]] = []
        for ep_idx in range(self.num_episodes):
            start = self.lerobot_dataset.episode_data_index["from"][ep_idx].item()
            end = self.lerobot_dataset.episode_data_index["to"][ep_idx].item()
            ep_len = end - start
            self.episodes.append((start, end, ep_len))

        # Filter episodes that are too short for the requested sequence
        span = (self.sequence_length - 1) * self.frame_skip + 1
        valid_episodes = [
            (i, s, e, l) for i, (s, e, l) in enumerate(self.episodes) if l >= span
        ]
        if not valid_episodes:
            raise ValueError(
                f"No episodes have >= {span} frames "
                f"(sequence_length={num_frames}, frame_skip={frame_skip}). "
                f"Max episode length: {max(l for _, _, l in self.episodes)}"
            )

        self.valid_episodes = valid_episodes
        self.num_failed_loads = 0

        log.info(
            f"LeRobot dataset {repo_id}: {self.num_episodes} episodes, "
            f"{len(self.valid_episodes)} valid (>= {span} frames), "
            f"cameras: {self.camera_keys}, fps: {self.fps}"
            + (f", wrist_blackout_prob: {self.wrist_blackout_prob}, "
               f"wrist_cameras: {self.wrist_camera_keys}"
               if self.wrist_blackout_prob > 0 else "")
        )

        self.preprocess = T.Compose([
            ToTensorVideo(),
            ResizePreprocess((video_size[0], video_size[1])),
        ])

        # Optional data augmentation to reduce overfitting
        # Matches cosmos-policy apply_image_aug() stronger=True:
        #   - Fixed 90% area crop (same region for all frames)
        #   - ColorJitter: brightness ±0.3, contrast ±0.4, saturation ±0.5, hue ±0.05
        self.augment_transform = None
        self.augment_crop_scale = (0.9, 0.9) if augment else None
        self.video_size = video_size
        if augment:
            self.augment_transform = T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.05)

    def __str__(self) -> str:
        return f"LeRobotVideoDatasetFlat({self.repo_id}, {len(self.valid_episodes)} episodes)"

    def __len__(self) -> int:
        return len(self.valid_episodes)

    def _load_episode_frames(
        self, ep_start: int, ep_end: int, ep_len: int
    ) -> tuple[torch.Tensor, str, list[int]]:
        """Load a sub-sequence of frames from one episode.

        Returns:
            frames: (T, C, H, W) uint8 tensor with cameras concatenated.
            caption: Task description string.
            frame_ids: [start_frame, end_frame, total_frames] metadata.
        """
        span = (self.sequence_length - 1) * self.frame_skip + 1
        max_start = ep_len - span
        local_start = np.random.randint(0, max_start + 1)

        # Global frame indices with frame_skip stride
        global_indices = list(range(
            ep_start + local_start,
            ep_start + local_start + span,
            self.frame_skip,
        ))

        # Load frames from each camera
        camera_frames = {key: [] for key in self.camera_keys}
        caption = None

        for global_idx in global_indices:
            item = self.lerobot_dataset[global_idx]

            if caption is None:
                caption = item.get("task", "")

            for key in self.camera_keys:
                # LeRobot returns images as float32 (C, H, W) in [0, 1]
                frame = item[key]  # (C, H, W) float32
                camera_frames[key].append(frame)

        # Stack each camera: (T, C, H, W)
        stacked = {
            key: torch.stack(camera_frames[key]) for key in self.camera_keys
        }

        # Combine cameras into flat layout
        combined = self._combine_cameras(stacked)

        # Convert from float [0,1] to uint8 [0,255]
        combined = torch.clamp(combined * 255.0, 0, 255).to(torch.uint8)

        frame_ids = [local_start, local_start + span, ep_len]
        return combined, caption or "", frame_ids

    def _maybe_blackout_wrist(
        self, stacked: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Randomly black out wrist camera views for augmentation.

        With wrist_blackout_prob split equally between two modes:
          - Half the prob: black out one random wrist camera
          - Half the prob: black out all wrist cameras
        E.g. wrist_blackout_prob=0.5 → 25% one wrist off, 25% both off, 50% normal.
        """
        if self.wrist_blackout_prob > 0 and self.wrist_camera_keys:
            r = np.random.random()
            half_prob = self.wrist_blackout_prob / 2.0
            if r < half_prob:
                # Black out one random wrist camera
                k = self.wrist_camera_keys[np.random.randint(len(self.wrist_camera_keys))]
                if k in stacked:
                    stacked[k] = torch.zeros_like(stacked[k])
            elif r < self.wrist_blackout_prob:
                # Black out all wrist cameras
                for k in self.wrist_camera_keys:
                    if k in stacked:
                        stacked[k] = torch.zeros_like(stacked[k])
        return stacked

    def _combine_cameras(
        self, stacked: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Combine multiple camera views into a single flat tensor.

        Args:
            stacked: Dict of camera_key -> (T, C, H, W) tensors.

        Returns:
            (T, C, H_combined, W_combined) tensor.
        """
        stacked = self._maybe_blackout_wrist(stacked)
        views = [stacked[k] for k in self.camera_keys]
        n_cams = len(views)

        if self.layout == "horizontal":
            # Simple horizontal concatenation: (T, C, H, W*N)
            return torch.cat(views, dim=-1)

        elif self.layout == "quad":
            # Flatten into 3 horizontal strips from a 2x2 quad:
            # top-left, top-right, bottom-left → cat horizontally as (H/2, W/2*3)
            T, C, H, W = views[0].shape
            half_h = H // 2
            half_w = W // 2

            # Crop each view to half-size
            crops = []
            for v in views[:3]:  # Use up to 3 cameras
                crops.append(v[:, :, :half_h, :half_w])

            # Pad if fewer than 3 cameras
            while len(crops) < 3:
                crops.append(torch.zeros(T, C, half_h, half_w, dtype=views[0].dtype))

            return torch.cat(crops, dim=-1)  # (T, C, H/2, W/2*3)

        else:
            raise ValueError(f"Unknown layout: {self.layout}. Use 'horizontal' or 'quad'.")

    def _getitem_inner(self, index: int) -> dict:
        ep_orig_idx, ep_start, ep_end, ep_len = self.valid_episodes[index]

        # Load frames: (T, C, H, W) uint8 with cameras already combined
        frames, caption, frame_ids = self._load_episode_frames(
            ep_start, ep_end, ep_len
        )

        # preprocess: ToTensorVideo (uint8 → float/255) + ResizePreprocess
        frames = self.preprocess(frames)
        # Random crop (same region for all frames in the video)
        if self.augment_crop_scale is not None:
            _, _, fh, fw = frames.shape
            scale = np.random.uniform(*self.augment_crop_scale)
            ch, cw = int(fh * scale), int(fw * scale)
            top = np.random.randint(0, fh - ch + 1)
            left = np.random.randint(0, fw - cw + 1)
            frames = frames[:, :, top:top+ch, left:left+cw]
            frames = torch.nn.functional.interpolate(
                frames, size=(self.video_size[0], self.video_size[1]),
                mode="bilinear", align_corners=False,
            )
        # Apply color jitter per-frame (frames is T,C,H,W float [0,1])
        if self.augment_transform is not None:
            frames = torch.stack([self.augment_transform(f) for f in frames])
        # Back to uint8 (same as VideoDatasetFlat._get_frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)

        # Rearrange from [T, C, H, W] to [C, T, H, W]
        video = frames.permute(1, 0, 2, 3)

        _, _, h, w = video.shape

        data = {
            "video": video,
            "ai_caption": caption,
            "fps": self.fps,
            "image_size": torch.tensor([h, w, h, w]),
            "num_frames": self.sequence_length,
            "padding_mask": torch.zeros(1, h, w),
        }
        return data

    def __getitem__(self, index: int) -> dict | Any:
        max_retries = 10
        last_error = None
        for attempt in range(max_retries):
            try:
                return self._getitem_inner(index)
            except Exception as e:
                last_error = e
                self.num_failed_loads += 1
                # Use print to ensure visibility in DataLoader worker processes
                import sys
                print(
                    f"[LeRobotDataset] Failed to load episode index={index} "
                    f"(attempt {attempt + 1}/{max_retries}, "
                    f"total failures: {self.num_failed_loads}): {e}\n"
                    f"{traceback.format_exc()}",
                    file=sys.stderr, flush=True,
                )
                # Try a different random episode next attempt
                index = np.random.randint(len(self.valid_episodes))

        raise RuntimeError(
            f"Failed to load any episode after {max_retries} retries "
            f"(total failures: {self.num_failed_loads}). "
            f"Last error: {last_error}"
        )


def get_generic_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    sampler: Optional[Any] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable] = None,
    **kwargs,
) -> DataLoader:
    """Create DataLoader with commonly used parameters."""
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )


def get_sampler(dataset) -> DistributedSampler:
    """Create a distributed sampler for the dataset."""
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )
