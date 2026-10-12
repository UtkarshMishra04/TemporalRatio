"""Video datasets for the video (DiT) loss.

``get_dataset_config(name, num_frames)`` returns the mixture used by the trainers'
``--video_dataset`` option:

* ``libero``: the LIBERO-Spatial/Object/Goal/10 demonstrations (``physical-intelligence/libero``)
  + LIBERO-90 (``IPEC-COMMUNITY/libero_90_no_noops_lerobot``), 50/50. The paper additionally mixed 10% failed
  rollouts; ``libero_with_failures`` does so when ``$TR_LIBERO_FAILURES_DIR`` points to a directory
  of ``videos/*.mp4`` + ``metas/*.txt``.
* ``libero_10``: ``physical-intelligence/libero`` only.
* ``abc130k_pp24``: the 22 ABC-130k tasks (``data/abc130k.py``), equal weight per task.
* ``abc130k_<group>``: one task group of ``data/abc130k.py``: the skills ``folding``, ``garment``,
  ``assembly``, ``pickplace`` and ``fold_pen_bottle`` (T-shirt folding, pen caps, bottle caps).

Cameras are tiled horizontally into one frame of ``video_size`` (H, W).
"""

import os
from dataclasses import dataclass, field, replace

from torch.utils.data import ConcatDataset

from temporal_ratio.data import abc130k
from temporal_ratio.data.video_loader import (
    LeRobotVideoDatasetFlat,
    LocalVideoDatasetFlat,
    create_dataloader,
    create_weighted_dataloader,
)


@dataclass
class VideoDatasetConfig:
    """Config for a single video dataset source."""
    repo_id: str
    num_frames: int = 17
    video_size: tuple[int, int] = (224, 448)
    camera_keys: list[str] | None = None
    layout: str = "horizontal"
    frame_skip: int = 1
    video_backend: str | None = None
    augment: bool = False
    root: str | None = None
    wrist_blackout_prob: float = 0.0
    cross_task_prob: float = 0.0

    def build(self) -> LeRobotVideoDatasetFlat:
        return LeRobotVideoDatasetFlat(
            repo_id=self.repo_id, num_frames=self.num_frames,
            video_size=self.video_size, camera_keys=self.camera_keys,
            layout=self.layout, frame_skip=self.frame_skip,
            video_backend=self.video_backend, augment=self.augment,
            root=self.root, wrist_blackout_prob=self.wrist_blackout_prob,
            cross_task_prob=self.cross_task_prob,
        )


@dataclass
class LocalVideoDatasetConfig:
    """Config for a local MP4 video dataset (videos/*.mp4 + metas/*.txt)."""
    dataset_dir: str
    num_frames: int = 17
    video_size: tuple[int, int] = (224, 448)
    frame_skip: int = 1

    def build(self) -> LocalVideoDatasetFlat:
        return LocalVideoDatasetFlat(
            dataset_dir=self.dataset_dir, num_frames=self.num_frames,
            video_size=self.video_size, frame_skip=self.frame_skip,
        )


@dataclass
class TrainingDataConfig:
    """Config for training data (possibly multiple datasets with weights)."""
    datasets: list[VideoDatasetConfig | LocalVideoDatasetConfig] = field(default_factory=list)
    weights: list[float] | None = None  # None = uniform, list = weighted sampling
    batch_size: int = 4
    num_workers: int = 8

    def build_dataloader(self, seed: int = 0):
        built = [d.build() for d in self.datasets]
        kwargs = dict(batch_size=self.batch_size, num_workers=self.num_workers, seed=seed)
        if len(built) == 1:
            return create_dataloader(built[0], **kwargs)
        elif self.weights is not None:
            return create_weighted_dataloader(built, self.weights, **kwargs)
        else:
            return create_dataloader(ConcatDataset(built), **kwargs)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

LIBERO_10 = VideoDatasetConfig(
    repo_id="physical-intelligence/libero",
    num_frames=17, video_size=(224, 448),
    layout="horizontal", augment=False,
)

LIBERO_90 = VideoDatasetConfig(
    repo_id="IPEC-COMMUNITY/libero_90_no_noops_lerobot",
    num_frames=17, video_size=(224, 448),
    layout="horizontal", augment=False, video_backend="pyav",
)


def _libero_failures() -> LocalVideoDatasetConfig:
    path = os.environ.get("TR_LIBERO_FAILURES_DIR")
    if not path:
        raise ValueError("libero_with_failures needs $TR_LIBERO_FAILURES_DIR (videos/*.mp4 + metas/*.txt)")
    return LocalVideoDatasetConfig(dataset_dir=path, num_frames=17, video_size=(224, 448))


def _abc130k(tasks: tuple[str, ...], batch_size: int) -> TrainingDataConfig:
    datasets = [
        VideoDatasetConfig(
            repo_id=abc130k.repo_id(task),
            num_frames=17, video_size=(176, 224 * 3),
            layout="horizontal", frame_skip=abc130k.FRAME_SKIP, video_backend="pyav",
        )
        for task in tasks
    ]
    return TrainingDataConfig(datasets=datasets, weights=[1.0 / len(tasks)] * len(tasks), batch_size=batch_size)


def _mixtures(batch_size: int) -> dict:
    mixtures = {
        "libero": lambda: TrainingDataConfig(datasets=[LIBERO_10, LIBERO_90], weights=[0.5, 0.5], batch_size=batch_size),
        "libero_with_failures": lambda: TrainingDataConfig(
            datasets=[LIBERO_10, LIBERO_90, _libero_failures()], weights=[0.45, 0.45, 0.10], batch_size=batch_size
        ),
        "libero_10": lambda: TrainingDataConfig(datasets=[LIBERO_10], batch_size=batch_size),
        "abc130k_pp24": lambda: _abc130k(abc130k.TASKS, batch_size),
    }
    for group, tasks in abc130k.TASK_GROUPS.items():
        mixtures[f"abc130k_{group}"] = lambda tasks=tasks: _abc130k(tasks, batch_size)
    return mixtures


def available() -> list[str]:
    return sorted(_mixtures(1))


def get_dataset_config(
    name: str, num_frames: int = 17, batch_size: int = 4, frame_skip: int | None = None
) -> TrainingDataConfig:
    """Video mixture ``name`` with clips of ``num_frames`` frames.

    ``frame_skip`` (the trainers' ``--video_frame_skip``) must match the dataset's own temporal
    subsampling, which is part of the dataset definition.
    """
    mixtures = _mixtures(batch_size)
    if name not in mixtures:
        raise ValueError(f"Unknown video dataset '{name}'. Available: {sorted(mixtures)}")
    config = mixtures[name]()
    datasets = [replace(d, num_frames=num_frames) for d in config.datasets]
    if frame_skip is not None:
        for d in datasets:
            if d.frame_skip != frame_skip:
                raise ValueError(
                    f"--video_frame_skip {frame_skip} does not match the frame skip of '{name}' ({d.frame_skip})"
                )
    return replace(config, datasets=datasets)
