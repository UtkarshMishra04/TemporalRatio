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

"""Callback to save raw training data videos at the start of training (step 0).

Useful for visually verifying the data pipeline (camera layout, frame skip,
resize, etc.) before waiting for the first model sampling checkpoint.
"""

import os

import torch
import wandb

from cosmos_video._src.imaginaire.model import ImaginaireModel
from cosmos_video._src.imaginaire.utils import distributed, log
from cosmos_video._src.imaginaire.utils.callback import Callback
from cosmos_video._src.imaginaire.visualize.video import save_img_or_video


class SaveDemoData(Callback):
    """Save raw training batch videos at step 0 for data pipeline verification.

    At the first training iteration, this callback saves each sample in the
    batch as an MP4 video (or JPG if single-frame) to a local directory and
    optionally logs them to W&B.

    Args:
        n_samples: Max number of samples from the batch to save. Default 4.
        fps: Frame rate for saved videos. Default 10.
    """

    def __init__(self, n_samples: int = 4, fps: int = 10) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.fps = fps
        self._done = False

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        config_job = self.config.job
        self.local_dir = f"{config_job.path_local}/SaveDemoData"
        if distributed.get_rank() == 0:
            os.makedirs(self.local_dir, exist_ok=True)
            log.info(f"SaveDemoData: will save demo videos to {self.local_dir}")

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if self._done:
            return

        # Only save on the very first iteration
        if iteration != 1:
            return

        self._done = True

        if distributed.get_rank() != 0:
            return

        video = data_batch.get("video")  # [B, C, T, H, W] uint8
        if video is None:
            log.warning("SaveDemoData: no 'video' key in data_batch, skipping.")
            return

        captions = data_batch.get("ai_caption", [None] * video.shape[0])
        n_save = min(self.n_samples, video.shape[0])

        log.info(f"SaveDemoData: saving {n_save} demo videos from first batch...")

        wandb_videos = {}
        for i in range(n_save):
            sample = video[i]  # [C, T, H, W]

            # Convert to float [0,1] for save_img_or_video
            # By training step end, the pipeline has normalized video to [-1, 1]
            if sample.dtype == torch.uint8:
                sample_float = sample.float() / 255.0
            else:
                # [-1, 1] → [0, 1]  (same as EveryNDrawSample.run_save)
                sample_float = (1.0 + sample.float().clamp(-1, 1)) / 2.0

            caption = captions[i] if i < len(captions) else ""
            if isinstance(caption, torch.Tensor):
                caption = ""

            fp = os.path.join(self.local_dir, f"demo_sample_{i:02d}")
            save_img_or_video(sample_float, fp, fps=self.fps)

            ext = ".jpg" if sample_float.shape[1] == 1 else ".mp4"
            log.info(f"  [{i}] saved: {fp}{ext}  caption: {caption!r}")

            if wandb.run:
                if ext == ".mp4":
                    wandb_videos[f"demo_data/sample_{i}"] = wandb.Video(
                        f"{fp}{ext}", caption=caption, fps=self.fps
                    )
                else:
                    wandb_videos[f"demo_data/sample_{i}"] = wandb.Image(
                        f"{fp}{ext}", caption=caption
                    )

        if wandb.run and wandb_videos:
            wandb.log(wandb_videos, step=0)
            log.info(f"SaveDemoData: logged {len(wandb_videos)} samples to W&B.")
