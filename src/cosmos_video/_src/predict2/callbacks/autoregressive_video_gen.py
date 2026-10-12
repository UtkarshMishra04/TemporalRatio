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

"""Callback for autoregressive video generation during training.

Generates a video, takes the last frame, uses it as conditioning to generate
the next video, repeats N times, concatenates all segments, and saves/uploads
the result for visualization.
"""

import os
from contextlib import nullcontext
from functools import partial

import torch
import torch.distributed as dist
import wandb
from einops import rearrange

from cosmos_video._src.imaginaire.callbacks.every_n import EveryN
from cosmos_video._src.imaginaire.model import ImaginaireModel
from cosmos_video._src.imaginaire.utils import distributed, log, misc
from cosmos_video._src.imaginaire.utils.parallel_state_helper import is_tp_cp_pp_rank0
from cosmos_video._src.imaginaire.visualize.video import save_img_or_video


class AutoregressiveVideoGen(EveryN):
    """Generate long videos autoregressively at configurable training intervals.

    At each trigger, takes a training batch sample, generates a video segment,
    then uses the last frame of that segment as conditioning to generate the
    next segment. Repeats for ``n_autoreg_steps`` iterations, concatenates
    all generated segments, and saves as MP4 + logs to W&B.

    Args:
        every_n: Run every N training iterations.
        n_autoreg_steps: Number of autoregressive generation steps. Default 5.
        guidance: Classifier-free guidance scale. Default 3.0.
        num_sampling_steps: Number of diffusion denoising steps. Default 35.
        n_samples: Number of samples from the batch to process. Default 1.
        fps: Frame rate for saved videos. Default 10.
        is_ema: Whether to run under EMA model weights. Default False.
    """

    def __init__(
        self,
        every_n: int,
        n_autoreg_steps: int = 5,
        guidance: float = 3.0,
        num_sampling_steps: int = 35,
        n_samples: int = 1,
        fps: int = 10,
        is_ema: bool = False,
        run_at_start: bool = False,
    ) -> None:
        super().__init__(every_n, step_size=1, run_at_start=run_at_start)
        self.n_autoreg_steps = n_autoreg_steps
        self.guidance = guidance
        self.num_sampling_steps = num_sampling_steps
        self.n_samples = n_samples
        self.fps = fps
        self.is_ema = is_ema
        self.name = self.__class__.__name__

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        config_job = self.config.job
        self.local_dir = f"{config_job.path_local}/{self.name}"
        if distributed.get_rank() == 0:
            os.makedirs(self.local_dir, exist_ok=True)
            log.info(f"AutoregressiveVideoGen: will save to {self.local_dir}")

    @torch.no_grad()
    def every_n_impl(self, trainer, model, data_batch, output_batch, loss, iteration):
        if self.is_ema:
            if not model.config.ema.enabled:
                return
            context = partial(model.ema_scope, "autoreg_video_gen")
        else:
            context = nullcontext

        tag = "ema" if self.is_ema else "reg"

        # Free training intermediate memory before generation
        torch.cuda.empty_cache()

        with context():
            self._generate_and_save(model, data_batch, iteration, tag)

        dist.barrier()
        torch.cuda.empty_cache()

    def _generate_and_save(
        self,
        model,
        data_batch: dict,
        iteration: int,
        tag: str,
    ) -> None:
        """Run autoregressive generation and save results."""
        # Limit batch to n_samples
        n = min(self.n_samples, data_batch["video"].shape[0])
        batch = {k: v[:n] if isinstance(v, torch.Tensor) else v for k, v in data_batch.items()}

        # Handle text embeddings computed online
        text_encoder_config = getattr(model.config, "text_encoder_config", None)
        if text_encoder_config is not None and text_encoder_config.compute_online:
            text_embeddings = model.text_encoder.compute_text_embeddings_online(
                batch, model.input_caption_key
            )
            batch["t5_text_embeddings"] = text_embeddings
            batch["t5_text_mask"] = torch.ones(
                text_embeddings.shape[0], text_embeddings.shape[1], device="cuda"
            )

        # Get video shape info from first batch for later steps
        # video is [B, C, T, H, W] uint8 at this point
        video_shape = batch["video"].shape  # [n, C, T, H, W]
        T_pixel = video_shape[2]
        H, W = video_shape[3], video_shape[4]

        all_segments_pixel = []  # Will collect decoded pixel segments

        log.info(
            f"AutoregressiveVideoGen: starting {self.n_autoreg_steps} steps, "
            f"video shape {list(video_shape)}, guidance={self.guidance}"
        )

        for step_idx in range(self.n_autoreg_steps):
            log.info(f"  Autoreg step {step_idx + 1}/{self.n_autoreg_steps}...")

            # Set conditioning: step 0 uses original batch's conditioning (from training data),
            # subsequent steps condition on 1 frame (the last frame of previous generation)
            if step_idx > 0:
                batch["num_conditional_frames"] = 1

            # Generate in latent space
            latent_samples = model.generate_samples_from_batch(
                batch,
                guidance=self.guidance,
                num_steps=self.num_sampling_steps,
            )

            # Decode to pixel space: [B, 3, T_pixel, H, W] in [-1, 1]
            decoded = model.decode(latent_samples)
            del latent_samples

            # Store this segment (skip first frame for steps > 0 since it's the conditioning frame)
            if step_idx == 0:
                all_segments_pixel.append(decoded.float().cpu())
            else:
                # Skip the conditioning frame (first frame) to avoid duplication
                all_segments_pixel.append(decoded[:, :, 1:].float().cpu())

            # Prepare next step: use last frame as conditioning
            if step_idx < self.n_autoreg_steps - 1:
                last_frame = decoded[:, :, -1:, :, :].clone()  # [B, 3, 1, H, W]
                del decoded

                # Build a new video: last_frame at position 0, zeros elsewhere
                new_video = torch.zeros(n, 3, T_pixel, H, W, dtype=last_frame.dtype, device=last_frame.device)
                new_video[:, :, 0:1, :, :] = last_frame
                del last_frame

                # Create fresh data_batch for next step
                batch = {
                    "video": new_video,
                    "is_preprocessed": True,  # Already in [-1, 1] float
                    "t5_text_embeddings": batch["t5_text_embeddings"],
                    "t5_text_mask": batch["t5_text_mask"],
                    "fps": batch["fps"],
                    "padding_mask": batch["padding_mask"],
                    "ai_caption": batch.get("ai_caption", ""),
                }
            else:
                del decoded

            torch.cuda.empty_cache()

        # Concatenate all segments along time: [B, 3, T_total, H, W]
        full_video = torch.cat(all_segments_pixel, dim=2)  # on CPU

        log.info(
            f"AutoregressiveVideoGen: generated full video shape {list(full_video.shape)}, "
            f"total frames: {full_video.shape[2]}"
        )

        # Save and log (rank 0 only)
        if not is_tp_cp_pp_rank0():
            return

        # Normalize from [-1, 1] to [0, 1] for saving
        full_video_01 = (1.0 + full_video.clamp(-1, 1)) / 2.0

        caption = data_batch.get("ai_caption", [""])[0] if isinstance(
            data_batch.get("ai_caption"), list
        ) else data_batch.get("ai_caption", "")
        if isinstance(caption, torch.Tensor):
            caption = ""

        wandb_videos = {}
        for i in range(full_video_01.shape[0]):
            sample = full_video_01[i]  # [C, T, H, W]
            fp = os.path.join(self.local_dir, f"{tag}_autoreg_iter{iteration:09d}_sample{i:02d}")
            save_img_or_video(sample, fp, fps=self.fps)

            ext = ".mp4"
            log.info(f"  Saved: {fp}{ext}  ({sample.shape[1]} frames)")

            if wandb.run:
                wandb_videos[f"autoreg_video/{tag}_sample_{i}"] = wandb.Video(
                    f"{fp}{ext}",
                    caption=f"step={iteration}, {self.n_autoreg_steps} autoreg steps, {caption}",
                    fps=self.fps,
                )

        if wandb.run and wandb_videos:
            wandb.log(wandb_videos, step=iteration)
            log.info(f"AutoregressiveVideoGen: logged {len(wandb_videos)} videos to W&B.")
