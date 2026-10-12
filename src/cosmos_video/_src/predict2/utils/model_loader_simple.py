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

import importlib
import os
from typing import Optional

import torch
from peft import LoraConfig, set_peft_model_state_dict

from cosmos_video._src.imaginaire.config import Config
from cosmos_video._src.imaginaire.flags import INTERNAL, SMOKE
from cosmos_video._src.imaginaire.lazy_config import instantiate
from cosmos_video._src.imaginaire.model import ImaginaireModel
from cosmos_video._src.imaginaire.utils import distributed, log, misc
from cosmos_video._src.imaginaire.utils.config_helper import get_config_module, override
from cosmos_video._src.imaginaire.utils.easy_io import easy_io
from cosmos_video._src.imaginaire.utils.fsdp_helper import hsdp_device_mesh


def load_model_from_checkpoint(
    experiment_name,
    s3_checkpoint_dir,
    config_file="cosmos_video/_src/predict2/configs/video2world/config.py",
    enable_fsdp=False,
    load_ema_to_reg=False,
    instantiate_ema=True,
    seed=0,
    local_cache_dir=None,
    override_cache: bool = False,
    experiment_opts: Optional[list[str]] = None,
    skip_load_model: bool = False,
    adapter_checkpoint_paths: Optional[list[str]] = None,
    cache_text_encoder: bool = False,
    to_device: Optional[str] = None,
    stop_rank_sync: bool = False,
):
    """
    Load model from checkpoint with optional multi-adapter support.

    Args:
        experiment_name: experiment name
        s3_checkpoint_dir: s3 path to iteration_model
        config_file: config file path
        enable_fsdp: enable fsdp
        load_ema_to_reg: load ema as regular model
        instantiate_ema: whether to instantiate EMA
        seed: random seed
        local_cache_dir: local cache directory, if None, do not cache
        override_cache: override cache, if True, override cache if local cache exists
        experiment_opts: experiment options
        skip_load_model: skip loading model weights
        adapter_checkpoint_paths: list of checkpoint paths for loading multiple adapters
            Supports both .pt and DCP checkpoint formats (auto-detected by file extension).
            Example:
                adapter_checkpoint_paths=[
                    "s3://bucket/exp1/checkpoints/model.pt",  # .pt format
                    "s3://bucket/exp2/checkpoints"  # DCP format
                ]
        cache_text_encoder: cache text encoder, if True, cache text encoder. This is default to False to avoid race condition if multiple nodes are running inference concurrently (e.g., running inference pipeline).

    Returns:
        model: loaded model
        config: config object
    """
    if experiment_opts is None:
        experiment_opts = []
    config_module = get_config_module(config_file)
    config = importlib.import_module(config_module).make_config()
    config = override(config, ["--", f"experiment={experiment_name}"] + experiment_opts)

    # Override checkpoint path if provided
    if s3_checkpoint_dir:
        log.info(f"Overriding config checkpoint path with: {s3_checkpoint_dir}")
        config.checkpoint.load_path = str(s3_checkpoint_dir)

    if load_ema_to_reg:
        config.model.config.ema.enabled = False

    if instantiate_ema is False and config.model.config.ema.enabled:
        config.model.config.ema.enabled = False

    # Skip validate() when running under torchrun / DDP — the broadcast inside
    # validate() conflicts with the already-initialised NCCL process group and
    # causes "Duplicate GPU detected" errors.
    if not (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1):
        config.validate()
    # Freeze the config so developers don't change it during training.
    config.freeze()  # type: ignore
    misc.set_random_seed(seed=seed, by_rank=True)
    # Initialize cuDNN.
    torch.backends.cudnn.deterministic = config.trainer.cudnn.deterministic
    torch.backends.cudnn.benchmark = config.trainer.cudnn.benchmark
    # Floating-point precision settings.
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True

    log.info(f"Loading model from {s3_checkpoint_dir}")

    config.model.config.fsdp_shard_size = 1
    with misc.timer("instantiate model"):
        model = instantiate(config.model)
        if to_device is not None:
            model.to(torch.device(to_device))
        # Convert the model parameters to bf16
        model.on_train_start()

    if not skip_load_model:
        # Load normally (single checkpoint)
        if stop_rank_sync:
            model = simple_load_model_from_checkpoint(
                model, config, s3_checkpoint_dir, load_ema_to_reg, local_cache_dir, override_cache, stop_rank_sync
            )
        else:
            raise NotImplementedError("Only simple_load_model_from_checkpoint with stop_rank_sync=True is implemented.")

    return model, config

def simple_load_model_from_checkpoint(
    model,
    config,
    s3_checkpoint_dir,
    load_ema_to_reg=False,
    local_cache_dir=None,
    override_cache: bool = False,
    stop_rank_sync: bool = False,
):

    from cosmos_video._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
    local_s3_ckpt_fp = get_checkpoint_path(s3_checkpoint_dir)

    log.info(f"Loading model cached locally from {local_s3_ckpt_fp}")
    # map_location: some fine-tuned checkpoints were saved with CUDA tensors.
    local_state_dict = easy_io.load(local_s3_ckpt_fp, weights_only=INTERNAL, map_location="cpu")

    model.load_state_dict(local_state_dict, strict=False)

    # Clear unused reserved memory from fp32
    torch.cuda.empty_cache()
    return model