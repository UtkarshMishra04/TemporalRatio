"""See _CONFIGS for the list of available configs."""

import abc
import dataclasses
import difflib
import logging
import os
import pathlib
from collections.abc import Sequence
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
import tyro
from typing_extensions import override

import temporal_ratio.data.abc130k as abc130k
import temporal_ratio.data.normalize as _normalize
import temporal_ratio.data.transforms as _transforms
import temporal_ratio.models.base as _model
import temporal_ratio.models.config as video_config
import temporal_ratio.policies.libero_policy as libero_policy
import temporal_ratio.policies.yam_policy as yam_policy
import temporal_ratio.training.optimizer as _optimizer
import temporal_ratio.training.weight_loaders as weight_loaders
from temporal_ratio._project_root import PROJECT_ROOT

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)
    obs_sequence_keys: Sequence[str] = ()

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    pad_progress: bool = False

    # Extra repo_ids to concatenate with the primary repo_id (for multi-task training).
    extra_repo_ids: tuple[str, ...] = ()

    # Frame skip for temporal subsampling (must match video world model training).
    # 1 = consecutive frames, 2 = every other frame, 4 = every 4th frame, etc.
    frame_skip: int = 1


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard video VLA models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    # Per-camera resize dimensions (height, width). Must match the video world
    # model's per-camera resolution so that flat concatenation yields the
    # expected IMAGE_SIZES entry.
    resize_height: int = 224
    resize_width: int = 224

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.VIDEO:
                assert isinstance(model_config, video_config.VideoVLAConfig)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(self.resize_height, self.resize_width),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )

@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    pad_progress: bool = False

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        asset_path = self.assets.assets_dir if assets_dirs is None else assets_dirs
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(asset_path), asset_id),
            use_quantile_norm=True,
            pad_progress=self.pad_progress,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        """Load ``<assets_dir>/<asset_id>/norm_stats.json``.

        ``assets_dir`` is ``TrainConfig.assets_dirs`` (``<repo>/assets/<config name>``). Released
        checkpoints carry their statistics in ``<checkpoint>/assets/``. ``$NORM_STATS_PATH`` (a
        norm_stats.json, or a directory containing exactly one, e.g. a checkpoint's ``assets/``)
        overrides the lookup.
        """
        override = os.environ.get("NORM_STATS_PATH")
        if override:
            path = _normalize.find(override)
            logging.info(f"Loading norm stats from $NORM_STATS_PATH: {path}/norm_stats.json")
            return _normalize.load(path)
        if asset_id is None:
            return None
        data_assets_dir = pathlib.Path(str(assets_dir)) / asset_id
        if not (data_assets_dir / "norm_stats.json").exists():
            # Fine for inference: the statistics come from the checkpoint or $NORM_STATS_PATH.
            logging.info(f"No norm stats at {data_assets_dir} (using the checkpoint's or $NORM_STATS_PATH)")
            return None
        logging.info(f"Loaded norm stats from {data_assets_dir}")
        return _normalize.load(data_assets_dir)


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)



@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                        "frame_index": "frame_index",
                        "episode_index": "episode_index"
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )







@dataclasses.dataclass(frozen=True)
class LeRobotABC130kDataConfig(DataConfigFactory):
    """Bimanual YAM data in the layout of the ABC-130k converter (on the ``master`` branch).

    Three cameras (``left_image`` = top, ``left_wrist_left``, ``right_wrist_left``), 14-D state and
    actions ``[left joints 1-6, left gripper, right joints 1-6, right gripper]``. Further tasks
    are concatenated via ``extra_repo_ids``.
    """

    extra_repo_ids: tuple[str, ...] = ()

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/left_image": "observation.images.left_image",
                        "observation/left_wrist_left": "observation.images.left_wrist_left",
                        "observation/right_wrist_left": "observation.images.right_wrist_left",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                        "frame_index": "frame_index",
                        "episode_index": "episode_index"
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[yam_policy.YAMInputs(model_type=model_config.model_type)],
            outputs=[yam_policy.YAMOutputs()],
        )
        # 168x224 frames are resized with padding to 176x224 per camera (16-divisible).
        model_transforms = ModelTransformFactory(resize_height=176, resize_width=224)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            extra_repo_ids=self.extra_repo_ids,
        )





@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "videoVLA"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    model: _model.BaseModelConfig = dataclasses.field(default_factory=video_config.VideoVLAConfig)

    # Video model setup (``temporal_ratio.models.video_model.VIDEO_SETUPS``): Cosmos experiment,
    # resolution of the tiled camera frame and the camera order.
    video_setup: str = "libero_flat_short"

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = str(PROJECT_ROOT / "assets")
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 96
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 4
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 5000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")

    def set_action_expert_config(self, action_expert_variant) -> "TrainConfig":
        """Set action_expert_config in the model based on the data config."""
        updated_model = self.model.set_action_expert_config(action_expert_variant)
        return dataclasses.replace(self, model=updated_model)

def _abc130k_config(name: str, repo_ids: tuple[str, ...], num_train_steps: int) -> "TrainConfig":
    return TrainConfig(
        name=name,
        model=video_config.VideoVLAConfig(
            observation_horizon=17, action_horizon=50, discrete_state_input=False, dtype="float32"
        ),
        video_setup="abc130k_flat_short",
        data=LeRobotABC130kDataConfig(
            repo_id=repo_ids[0],
            extra_repo_ids=repo_ids[1:],
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
                obs_sequence_keys=(
                    "observation.images.left_image",
                    "observation.images.left_wrist_left",
                    "observation.images.right_wrist_left",
                ),
                frame_skip=abc130k.FRAME_SKIP,
            ),
        ),
        batch_size=4,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=500, peak_lr=5e-4, decay_steps=10_000, decay_lr=5e-5),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=None,
        num_train_steps=num_train_steps,
        save_interval=1000,
    )


# Named robot/dataset configs. Inference YAMLs reference them as ``policy_config``.
_CONFIGS = [
    # LIBERO (physical-intelligence/libero): two 224x224 cameras tiled to 224x448, 17-step chunks.
    TrainConfig(
        name="videoVLA_libero_short",
        model=video_config.VideoVLAConfig(
            observation_horizon=17,
            action_horizon=17,
            discrete_state_input=False,
            dtype="float32",
        ),
        video_setup="libero_flat_short",
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("actions",),
                obs_sequence_keys=("image", "wrist_image"),
            ),
            extra_delta_transform=False,
        ),
        batch_size=16,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-4,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=None,
        num_train_steps=10_000,
        save_interval=500,
    ),
    # ABC-130k (bimanual YAM): the 22 tasks shared with the "pp24" training set, 50-step chunks.
    _abc130k_config("videoVLA_abc130k_pp24_short", abc130k.repo_ids(), num_train_steps=50_000),
    # ... and one config per task group (skills, and folding + pen caps + bottle caps).
    *[
        _abc130k_config(f"videoVLA_abc130k_{group}_short", abc130k.repo_ids(group), num_train_steps=20_000)
        for group in abc130k.TASK_GROUPS
    ],
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
