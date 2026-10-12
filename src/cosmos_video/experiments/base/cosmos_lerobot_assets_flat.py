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

"""Cosmos-Predict2.5 2B Video2World experiment configs used by this repository.

``temporal_ratio.models.video_model.VIDEO_SETUPS`` selects one of these experiments to build the
video model (architecture, tokenizer, conditioning); the trainers in ``temporal_ratio.training``
bring their own data loaders. The data loaders below are only used when training the video
model with the Cosmos trainer directly.

* ``predict2_video2world_training_2b_cosmos_lerobot_libero_flat_short``: LIBERO, 2 cameras.
* ``predict2_video2world_training_2b_cosmos_lerobot_yam_flat_short``: bimanual YAM (ABC-130k), 3 cameras.
"""

from hydra.core.config_store import ConfigStore
from torch.utils.data import ConcatDataset

from cosmos_video._src.imaginaire.lazy_config import LazyCall as L
from cosmos_video._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_video._src.predict2.callbacks.save_demo_data import SaveDemoData
from cosmos_video._src.predict2.datasets.local_datasets.dataset_video_lerobot import LeRobotVideoDatasetFlat, get_generic_dataloader, get_sampler
from cosmos_video._src.predict2.datasets.local_datasets.dataset_video_libero import VideoDataset as LiberoVideoDataset, get_generic_dataloader as get_generic_dataloader_libero, get_sampler as get_sampler_libero
from cosmos_video.config import MODEL_CHECKPOINTS, ModelKey

DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]


# ---------------------------------------------------------------------------
# Training config automation: LR scaling + max_iter from dataset size
# ---------------------------------------------------------------------------
# Dataset metadata: {name: (num_episodes, avg_episode_length, frame_skip)}
DATASET_INFO = {
    "libero":  (1693 + 3921, 162, 1),  # libero_10 (1693) + libero_90 (3921)
    "yam":     (92,    448, 4),
}
NUM_FRAMES_PER_SAMPLE = 17


def _detect_num_gpus() -> int:
    """Auto-detect number of GPUs from environment (set by torchrun/slurm)."""
    import os
    # torchrun sets WORLD_SIZE; slurm sets SLURM_NTASKS or SLURM_GPUS_ON_NODE * SLURM_NNODES
    for var in ("WORLD_SIZE", "SLURM_NTASKS"):
        val = os.environ.get(var)
        if val is not None:
            return int(val)
    # Fallback: count visible CUDA devices
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        return len(cuda_visible.split(","))
    return 8  # safe default


def compute_training_params(
    dataset_name: str,
    batch_size_per_gpu: int,
    num_gpus: int | None = None,
    base_lr: float = 2 ** (-14.5),
    target_full_passes: int = 15,
    min_iter: int = 5000,
    max_iter_cap: int = 80000,
) -> dict:
    """Compute LR and max_iter (target full-coverage passes).

    A "full-coverage pass" = enough samples to statistically cover every frame
    in every episode once (accounting for frame_skip and num_frames_per_sample).

    Args:
        num_gpus: Number of GPUs. None = auto-detect from WORLD_SIZE / CUDA_VISIBLE_DEVICES.
        base_lr: Reference LR (used directly, no batch scaling — matches MP4 pipeline).
        target_full_passes: How many times to cover every frame in every episode.
        min_iter: Floor on max_iter (diffusion models need enough steps to converge).
        max_iter_cap: Ceiling on max_iter to keep training time bounded.
    """
    if num_gpus is None:
        num_gpus = _detect_num_gpus()
    num_episodes, avg_ep_len, frame_skip = DATASET_INFO[dataset_name]
    effective_batch = batch_size_per_gpu * num_gpus

    # Use base_lr directly — no batch scaling.
    # The MP4 pipeline (cosmos_libero_both_assets_flat) uses lr=2**(-14.5) without
    # scaling, and that setup does not overfit. The previous 16x linear scaling
    # (effective_batch=128 / ref=8) was causing rapid memorisation.
    lr = base_lr

    # Samples needed per episode to cover all its (subsampled) frames
    effective_ep_len = avg_ep_len / frame_skip
    samples_per_episode = max(1, effective_ep_len / NUM_FRAMES_PER_SAMPLE)

    # Total steps for target full-coverage passes, clamped to [min_iter, max_iter_cap]
    raw_iter = target_full_passes * num_episodes * samples_per_episode / effective_batch
    max_iter = int(min(max_iter_cap, max(min_iter, raw_iter)))
    # Round to nearest 1000
    max_iter = max(1000, round(max_iter / 1000) * 1000)

    # Save ~10 checkpoints per run (min 500)
    save_iter = max(500, round(max_iter / 10 / 500) * 500)

    # Warmup = ~5% of training, capped at [500, 3000]
    warm_up_steps = min(3000, max(500, max_iter // 20))

    return dict(lr=lr, max_iter=max_iter, save_iter=save_iter, warm_up_steps=warm_up_steps)



# ---------------------------------------------------------------------------
# LIBERO dataset (libero_10 + libero_90, 2 cameras: image + wrist_image)
# ---------------------------------------------------------------------------
example_lerobot_libero_10_short = L(LeRobotVideoDatasetFlat)(
    repo_id="physical-intelligence/libero",
    num_frames=17,
    video_size=(224, 224 * 2),
    camera_keys=None,  # Auto-detect from dataset
    layout="horizontal",  # 2 cameras → horizontal concat
    augment=False,
)

example_lerobot_libero_90_short = L(LeRobotVideoDatasetFlat)(
    repo_id="IPEC-COMMUNITY/libero_90_no_noops_lerobot",
    num_frames=17,
    video_size=(224, 224 * 2),
    camera_keys=None,
    layout="horizontal",
    augment=False,
)

example_lerobot_libero_short = L(ConcatDataset)(
    datasets=[example_lerobot_libero_10_short, example_lerobot_libero_90_short],
)

dataloader_train_lerobot_libero_short = L(get_generic_dataloader)(
    dataset=example_lerobot_libero_short,
    sampler=L(get_sampler)(dataset=example_lerobot_libero_short),
    batch_size=4,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)


# ---------------------------------------------------------------------------
# YAM dataset (3 cameras: left_image, left_wrist_left, right_wrist_left)
# ---------------------------------------------------------------------------
example_lerobot_yam_short = L(LeRobotVideoDatasetFlat)(
    repo_id="abc130k/fold_and_stack_the_t_shirts",  # scripts/data/convert_abc130k_to_lerobot.py
    num_frames=17,
    video_size=(176, 224 * 3),  # 3 cameras of 176x224, tiled horizontally
    camera_keys=None,  # Auto-detect
    layout="horizontal",  # 3 cameras → horizontal flatten
    frame_skip=3,
    video_backend="pyav",  # av1 codec not supported by torchcodec
)

dataloader_train_lerobot_yam_short = L(get_generic_dataloader)(
    dataset=example_lerobot_yam_short,
    sampler=L(get_sampler)(dataset=example_lerobot_yam_short),
    batch_size=4,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)


# ---------------------------------------------------------------------------
# Experiment: LIBERO
# ---------------------------------------------------------------------------
_libero_2b = compute_training_params("libero", batch_size_per_gpu=4, base_lr=2 ** (-17), target_full_passes=5000)

_libero_ood_val_dir = None
example_libero_ood_val = L(LiberoVideoDataset)(
    dataset_dir=_libero_ood_val_dir,
    num_frames=17,
    video_size=(224, 448),
)
dataloader_val_libero_ood = L(get_generic_dataloader_libero)(
    dataset=example_libero_ood_val,
    sampler=L(get_sampler_libero)(dataset=example_libero_ood_val),
    batch_size=4,
    drop_last=False,
    num_workers=4,
    pin_memory=True,
)

predict2_video2world_training_2b_cosmos_lerobot_libero_flat_short = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_cosmos_lerobot_libero_flat_short",
    ),
    dataloader_train=dataloader_train_lerobot_libero_short,
    dataloader_val=dataloader_val_libero_ood,
    checkpoint=dict(
        save_iter=2000,
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    optimizer=dict(
        lr=_libero_2b["lr"],
        weight_decay=0.001,
    ),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[_libero_2b["warm_up_steps"]],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=80000,
        run_validation=True,
        validation_iter=2000,
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=2000, save_s3=False),
            every_n_sample_ema=dict(every_n=2000, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            save_demo_data=L(SaveDemoData)(n_samples=4, fps=10),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
)


# ---------------------------------------------------------------------------
# Experiment: YAM
# ---------------------------------------------------------------------------
_yam_2b = compute_training_params("yam", batch_size_per_gpu=4, base_lr=2 ** (-14.5))
predict2_video2world_training_2b_cosmos_lerobot_yam_flat_short = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_cosmos_lerobot_yam_flat_short",
    ),
    dataloader_train=dataloader_train_lerobot_yam_short,
    checkpoint=dict(
        save_iter=_yam_2b["save_iter"],
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    optimizer=dict(
        lr=_yam_2b["lr"],
        weight_decay=0.001,
    ),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[_yam_2b["warm_up_steps"]],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=_yam_2b["max_iter"],
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=5000, save_s3=False),
            every_n_sample_ema=dict(every_n=5000, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            save_demo_data=L(SaveDemoData)(n_samples=4, fps=10),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
)


cs = ConfigStore.instance()

for _item in [
    predict2_video2world_training_2b_cosmos_lerobot_libero_flat_short,
    predict2_video2world_training_2b_cosmos_lerobot_yam_flat_short,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]  # noqa: RUF015
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
