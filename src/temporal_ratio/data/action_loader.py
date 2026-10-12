from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

from temporal_ratio.data import video_decode as _video_decode

_video_decode.install()
import numpy as np
import torch
from pathlib import Path

import temporal_ratio.models.base as _model
import temporal_ratio.config.registry as _config
import temporal_ratio.data.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        # Retry on AssertionError from torchcodec video decoding tolerance issues
        import random
        for attempt in range(5):
            try:
                return self._transform(self._dataset[index])
            except AssertionError:
                if attempt == 4:
                    raise
                index = random.randint(0, len(self._dataset) - 1)

    def get_episode_info(self, episode_idx: SupportsIndex):
        if hasattr(self._dataset, "get_episode_info"):
            return self._dataset.get_episode_info(episode_idx)
        elif hasattr(self._dataset, "meta"):
            return self._dataset.meta.episodes[episode_idx]
        else:
            raise AttributeError("The underlying dataset does not have 'get_episode_info' method or 'meta' attribute.")

    def __len__(self) -> int:
        return len(self._dataset)




class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, observation_horizon: int, model_config: _model.BaseModelConfig,
    episode_ids_per_repo: dict[str, list[int]] | None = None,
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)

    frame_skip = getattr(data_config, "frame_skip", 1)
    delta_timestamps = {}
    for key in data_config.action_sequence_keys:
        hz = action_horizon
        delta_timestamps[key] = [t / dataset_meta.fps for t in range(hz)]

    for key in data_config.obs_sequence_keys:
        hz = observation_horizon
        delta_timestamps[key] = [t * frame_skip / dataset_meta.fps for t in range(hz)]

    _ep_filter = episode_ids_per_repo.get(data_config.repo_id) if episode_ids_per_repo else None
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps=delta_timestamps,
        episodes=_ep_filter,
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    # Handle extra_repo_ids for multi-task training (e.g. GR1 with 24 tasks)
    extra_repo_ids = getattr(data_config, "extra_repo_ids", ())
    if extra_repo_ids:
        from torch.utils.data import ConcatDataset
        import logging
        _log = logging.getLogger(__name__)

        _log.info(f"Loading {len(extra_repo_ids)} extra datasets sequentially...")
        all_datasets = [dataset]
        for i, rid in enumerate(extra_repo_ids):
            meta = lerobot_dataset.LeRobotDatasetMetadata(rid)
            _ep_f = episode_ids_per_repo.get(rid) if episode_ids_per_repo else None
            ds = lerobot_dataset.LeRobotDataset(rid, delta_timestamps=delta_timestamps, episodes=_ep_f)
            if data_config.prompt_from_task:
                ds = TransformedDataset(ds, [_transforms.PromptFromLeRobotTask(meta.tasks)])
            _log.info(f"  [{i+1}/{len(extra_repo_ids)}] Loaded {rid} ({len(ds)} frames)")
            all_datasets.append(ds)

        dataset = ConcatDataset(all_datasets)
        dataset._dataset_lengths = [len(d) for d in all_datasets]
        _log.info(f"All {len(extra_repo_ids)} extra datasets loaded")

        # Equal weight per task: each task gets 1/n_tasks share regardless of
        # episode count or skill category.
        n_tasks = len(dataset._dataset_lengths)
        sample_weights = []
        for ds_len in dataset._dataset_lengths:
            w = (1.0 / n_tasks) * (1.0 / max(ds_len, 1))
            sample_weights.extend([w] * ds_len)
        dataset._sample_weights = sample_weights
        _log.info(f"Task-balanced weights: {n_tasks} tasks, equal share per task")

    return dataset




def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False, additional_transforms: list[_transforms.DataTransformFn] = []) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
            *additional_transforms,
        ],
    )




def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
    flat_frames: bool = False,
    additional_transforms: list[_transforms.DataTransformFn] = [],
    episode_ids_per_repo: dict[str, list[int]] | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        observation_horizon=config.model.observation_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        flat_frames=flat_frames,
        additional_transforms=additional_transforms,
        episode_ids_per_repo=episode_ids_per_repo,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    observation_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    flat_frames: bool = False,
    additional_transforms: list[_transforms.DataTransformFn] = [],
    episode_ids_per_repo: dict[str, list[int]] | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, observation_horizon, model_config, episode_ids_per_repo=episode_ids_per_repo)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, additional_transforms=additional_transforms)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()

            # Use skill-balanced weighted sampler if available
            if hasattr(dataset, "_sample_weights"):
                import torch as _torch
                weights = _torch.tensor(dataset._sample_weights, dtype=_torch.double)
                n_samples = len(weights) // world_size
                sampler = _torch.utils.data.WeightedRandomSampler(
                    weights, num_samples=n_samples, replacement=True,
                )
                logging.info(f"Using skill-balanced WeightedRandomSampler ({n_samples} samples/rank)")
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=shuffle,
                    drop_last=True,
                )

            # Allow overriding per-rank local batch sizes using an environment variable
            # Format: PER_RANK_BATCH_SIZES="b0,b1,b2,..." (length must equal world_size)
            per_rank_env = os.environ.get("PER_RANK_BATCH_SIZES")
            if per_rank_env:
                try:
                    per_rank_sizes = [int(x) for x in per_rank_env.split(",")]
                    if len(per_rank_sizes) == world_size:
                        local_batch_size = per_rank_sizes[rank]
                    else:
                        logging.warning(
                            "PER_RANK_BATCH_SIZES length (%d) != world_size (%d). Falling back to uniform split.",
                            len(per_rank_sizes),
                            world_size,
                        )
                        local_batch_size = batch_size # // world_size
                except Exception:
                    logging.exception("Failed to parse PER_RANK_BATCH_SIZES. Falling back to uniform split.")
                    local_batch_size = batch_size # // world_size
            else:
                # Default: uniform split across ranks
                local_batch_size = batch_size # // world_size
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size # // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
        setup="robocasa" if "robocasa" in data_config.repo_id else "libero",
        pad_progress=data_config.pad_progress,
        flat_frames=flat_frames
    )

    return DataLoaderImpl(data_config, data_loader)



class WanTransformedDataset(Dataset):
    def __init__(self, dataset, pad_progress=False, flat_frames=False):
        self.dataset = dataset
        # self.embedding_dict = embedding_dict
        self.pad_progress = pad_progress
        self.flat_frames = flat_frames
        self.image_buffer = None
        self.count = 0

    def __getitem__(self, index):
        # This now runs in PARALLEL background workers
        self.count += 1
        
        # Verbose log every 500 samples per worker
        if self.count % 500 == 0:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info else "Main"
            print(f"DEBUG: [Worker {worker_id}] processed {self.count} samples. Current index: {index}")

        batch = self.dataset[index]
        
        if "left_base_0_rgb" in batch["image"]:
            # Robocasa dataset

            T, H, W, C = batch["image"]["left_base_0_rgb"].shape

            if self.flat_frames:
                combined_wan_images_latents = np.concatenate([
                    batch["image"]["left_base_0_rgb"],
                    batch["image"]["right_base_0_rgb"],
                    batch["image"]["left_wrist_0_rgb"]
                ], axis=2)
            else:
                if self.image_buffer is None:
                    self.image_buffer = np.zeros((T, H*2, W*2, C), dtype=batch["image"]["left_base_0_rgb"].dtype)   

                combined_wan_images_latents = self.image_buffer.copy()
                combined_wan_images_latents[:, :H, :W, :] = batch["image"]["left_base_0_rgb"]
                combined_wan_images_latents[:, :H, W:, :] = batch["image"]["right_base_0_rgb"]
                combined_wan_images_latents[:, H:, :W, :] = batch["image"]["left_wrist_0_rgb"]

            if self.pad_progress and not self.flat_frames:
                # prompt modification
                episode_idx = batch["episode_index"]
                start_index = batch["frame_index"]
                end_index = start_index + batch["actions"].shape[0] - 1
                episode_length = self.dataset.get_episode_info(episode_idx)
                current_prompt = batch["prompt"]
                new_text = current_prompt + f" Current progress is from {start_index} to {end_index} out of {episode_length} frames."
                batch["prompt"] = new_text

                bar_width = W
                bar_height = H

                progress_bar = np.ones((T, bar_height, bar_width, C), dtype=combined_wan_images_latents.dtype)*255
                start_h = int(float(start_index / episode_length) * bar_height)
                end_h = int(float(end_index / episode_length) * bar_height)
                
                # Ensure indices are within bounds
                start_h = max(0, min(bar_height, start_h))
                end_h = max(0, min(bar_height, end_h))

                # Set the corresponding rows to black
                if end_h > start_h:
                    progress_bar[:, start_h:end_h, :, :] = 0

                combined_wan_images_latents[:, -bar_height:, -bar_width:, :] = progress_bar

            # wan_tokenized_prompt = np.array([self.embedding_dict[prompt] for prompt in batch["prompt"]])

            # batch["wan_tokenized_prompt"] = wan_tokenized_prompt
            batch["wan_images_latents"] = combined_wan_images_latents
            # batch["prompt"] = batch["prompt"].tolist()  # Convert from numpy array of strings to list of strings

            batch.pop("image")
        elif "ego_view_0_rgb" in batch["image"]:
            # GR1 dataset - single ego_view camera (224x224)
            T, H, W, C = batch["image"]["ego_view_0_rgb"].shape
            combined_wan_images_latents = batch["image"]["ego_view_0_rgb"]

            batch["wan_images_latents"] = combined_wan_images_latents
            batch.pop("image")
        else:
            # Libero dataset - need to add WAN embeddings and rearrange images
            T, H, W, C = batch["image"]["base_0_rgb"].shape

            if self.flat_frames:
                combined_wan_images_latents = np.concatenate([
                    batch["image"]["base_0_rgb"],
                    batch["image"]["left_wrist_0_rgb"]
                ], axis=2)
            else:
                combined_wan_images_latents = np.zeros((T, H*2, W*2, C), dtype=batch["image"]["base_0_rgb"].dtype)
                combined_wan_images_latents[:, :H, :W, :] = batch["image"]["base_0_rgb"]
                combined_wan_images_latents[:, H:, :W, :] = batch["image"]["left_wrist_0_rgb"]

            if self.pad_progress and not self.flat_frames:
                # prompt modification
                episode_idx = batch["episode_index"]
                start_index = batch["frame_index"]
                end_index = start_index + batch["actions"].shape[0] - 1
                episode_length = self.dataset.get_episode_info(episode_idx)
                current_prompt = batch["prompt"]
                new_text = current_prompt + f" Current progress is from {start_index} to {end_index} out of {episode_length} frames."
                batch["prompt"] = new_text

                bar_width = W
                bar_height = H

                progress_bar = np.ones((T, bar_height, bar_width, C), dtype=combined_wan_images_latents.dtype)*255
                start_h = int(float(start_index / episode_length) * bar_height)
                end_h = int(float(end_index / episode_length) * bar_height)
                
                # Ensure indices are within bounds
                start_h = max(0, min(bar_height, start_h))
                end_h = max(0, min(bar_height, end_h))

                # Set the corresponding rows to black
                if end_h > start_h:
                    progress_bar[:, start_h:end_h, :, :] = 0

                combined_wan_images_latents[:, -bar_height:, -bar_width:, :] = progress_bar
                
            # wan_tokenized_prompt = np.array([self.embedding_dict[prompt] for prompt in batch["prompt"]])

            # batch["wan_tokenized_prompt"] = wan_tokenized_prompt
            batch["wan_images_latents"] = combined_wan_images_latents
            # batch["prompt"] = batch["prompt"].tolist()  # Convert from numpy array of strings to list of strings

            batch.pop("image")
        
        # Move the embedding lookup here
        # if "prompt" in batch:
        #     batch["wan_tokenized_prompt"] = self.embedding_dict[batch["prompt"]]
            
        return batch

    def __len__(self):
        return len(self.dataset)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
        setup: str = "libero",
        pad_progress: bool = False,
        flat_frames: bool = False
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")


        # embedding_dict = {}
        # for prompt, path in embedding_path_dict.items():
        #     # mmap_mode='r' avoids copying data into RAM; all 8 GPU processes on the
        #     # same node share the same physical pages via the OS page cache.
        #     # This reduces per-node memory from 8×46GB=368GB to ~46GB shared.
        #     embedding_dict[prompt] = np.load(path, mmap_mode="r")

        # self.embedding_dict = embedding_dict

        dataset = WanTransformedDataset(
            dataset=dataset,
            # embedding_dict=embedding_dict,
            pad_progress=pad_progress,
            flat_frames=flat_frames,
        )

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.

                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    # if "left_base_0_rgb" in batch["image"]:
                    #     # Robocasa dataset

                    #     B, T, H, W, C = batch["image"]["left_base_0_rgb"].shape

                    #     combined_wan_images_latents = np.zeros((B, T, H*2, W*2, C), dtype=batch["image"]["left_base_0_rgb"].dtype)
                    #     combined_wan_images_latents[:, :, :H, :W, :] = batch["image"]["left_base_0_rgb"]
                    #     combined_wan_images_latents[:, :, :H, W:, :] = batch["image"]["right_base_0_rgb"]
                    #     combined_wan_images_latents[:, :, H:, :W, :] = batch["image"]["left_wrist_0_rgb"]
    
                    #     if self.pad_progress:
                    #         # prompt modification
                    #         for i in range(len(batch["prompt"])):
                    #             episode_idx = batch["episode_index"][i]
                    #             start_index = batch["frame_index"][i]
                    #             end_index = start_index + batch["actions"][i].shape[0] - 1
                    #             episode_length = self._data_loader.dataset.get_episode_info(episode_idx)
                    #             current_prompt = batch["prompt"][i]
                    #             new_text = current_prompt + f" Current progress is from {start_index} to {end_index} out of {episode_length} frames."
                    #             batch["prompt"][i] = new_text

                    #             bar_width = W
                    #             bar_height = H

                    #             progress_bar = np.ones((T, bar_height, bar_width, C), dtype=combined_wan_images_latents.dtype)*255
                    #             start_h = int(float(start_index / episode_length) * bar_height)
                    #             end_h = int(float(end_index / episode_length) * bar_height)
                                
                    #             # Ensure indices are within bounds
                    #             start_h = max(0, min(bar_height, start_h))
                    #             end_h = max(0, min(bar_height, end_h))

                    #             # Set the corresponding rows to black
                    #             if end_h > start_h:
                    #                 progress_bar[:, start_h:end_h, :, :] = 0

                    #             combined_wan_images_latents[i, :, -bar_height:, -bar_width:, :] = progress_bar

                    #     wan_tokenized_prompt = np.array([self.embedding_dict[prompt] for prompt in batch["prompt"]])

                    #     batch["wan_tokenized_prompt"] = wan_tokenized_prompt
                    #     batch["wan_images_latents"] = combined_wan_images_latents
                    #     batch["prompt"] = batch["prompt"].tolist()  # Convert from numpy array of strings to list of strings

                    #     batch.pop("image")

                    #     yield jax.tree.map(
                    #         lambda x: x if not hasattr(x, "shape") else torch.as_tensor(x),
                    #         batch
                    #     )
                    # else:
                    #     # Libero dataset - need to add WAN embeddings and rearrange images
                    #     B, T, H, W, C = batch["image"]["base_0_rgb"].shape

                    #     combined_wan_images_latents = np.zeros((B, T, H*2, W*2, C), dtype=batch["image"]["base_0_rgb"].dtype)
                    #     combined_wan_images_latents[:, :, :H, :W, :] = batch["image"]["base_0_rgb"]
                    #     combined_wan_images_latents[:, :, H:, :W, :] = batch["image"]["left_wrist_0_rgb"]

                    #     if self.pad_progress:
                    #         # prompt modification
                    #         for i in range(len(batch["prompt"])):
                    #             episode_idx = batch["episode_index"][i]
                    #             start_index = batch["frame_index"][i]
                    #             end_index = start_index + batch["actions"][i].shape[0] - 1
                    #             episode_length = self._data_loader.dataset.get_episode_info(episode_idx)
                    #             current_prompt = batch["prompt"][i]
                    #             new_text = current_prompt + f" Current progress is from {start_index} to {end_index} out of {episode_length} frames."
                    #             batch["prompt"][i] = new_text

                    #             bar_width = W
                    #             bar_height = H

                    #             progress_bar = np.ones((T, bar_height, bar_width, C), dtype=combined_wan_images_latents.dtype)*255
                    #             start_h = int(float(start_index / episode_length) * bar_height)
                    #             end_h = int(float(end_index / episode_length) * bar_height)
                                
                    #             # Ensure indices are within bounds
                    #             start_h = max(0, min(bar_height, start_h))
                    #             end_h = max(0, min(bar_height, end_h))

                    #             # Set the corresponding rows to black
                    #             if end_h > start_h:
                    #                 progress_bar[:, start_h:end_h, :, :] = 0

                    #             combined_wan_images_latents[i, :, -bar_height:, -bar_width:, :] = progress_bar
                            
                    #     wan_tokenized_prompt = np.array([self.embedding_dict[prompt] for prompt in batch["prompt"]])

                    #     batch["wan_tokenized_prompt"] = wan_tokenized_prompt
                    #     batch["wan_images_latents"] = combined_wan_images_latents
                    #     batch["prompt"] = batch["prompt"].tolist()  # Convert from numpy array of strings to list of strings

                    #     batch.pop("image")

                    # wan_tokenized_prompt = np.array([self.embedding_dict[prompt] for prompt in batch["prompt"]])

                    batch["wan_tokenized_prompt"] = None # wan_tokenized_prompt
                    batch["prompt"] = batch["prompt"].tolist()

                    yield jax.tree.map(
                        lambda x: x if not hasattr(x, "shape") else torch.as_tensor(x),
                        batch
                    )


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    rank = os.environ.get("RANK", "0")
    print(f"==> [Rank {rank}] Worker {worker_id} initialized and starting to load data.")



class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"], batch["prompt"]
