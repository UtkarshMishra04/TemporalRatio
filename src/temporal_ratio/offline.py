"""Offline (open-loop) evaluation of a checkpoint on recorded LeRobot episodes.

Every ``replan_every`` frames of an episode the policy receives the recorded observation and
prompt; its action chunk is compared with the recorded actions. Per step this records the action
error, the Temporal Ratio of ``tr_log_layers`` and timings; with ``generate_video`` it also writes
the predicted future frames next to the recorded ones. Errors are also reported in units of the
per-dimension action standard deviation of the evaluated (held-out) dataset (``*_std`` metrics).

Datasets, episodes and the query budget default to the config's ``offline`` section. Outputs
(``out_dir``): ``report.json`` (config, per-query metrics, per-dataset summaries) and, per dataset
(``<owner>__<name>/``), ``episode_<e>.npz`` (predicted / recorded actions) and
``episode_<e>_step_<t>.mp4`` videos.
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
from typing import Any

import numpy as np

from temporal_ratio.config.schema import InferenceExperiment, as_dict

log = logging.getLogger(__name__)


def std_errors(pred: np.ndarray, gt: np.ndarray, std: np.ndarray, first_k: int | None = None) -> dict[str, float]:
    """Errors of a predicted action chunk in units of the per-dimension standard deviation ``std``;
    with ``first_k``, also over the first ``first_k`` actions."""
    e = (pred - gt) / std
    out = {
        "action_mse_std": float(np.mean(e**2)),
        "action_mae_std": float(np.mean(np.abs(e))),
        "first_action_mae_std": float(np.mean(np.abs(e[0]))),
    }
    if first_k:
        out["first_k_mse_std"] = float(np.mean(e[:first_k] ** 2))
        out["first_k_mae_std"] = float(np.mean(np.abs(e[:first_k])))
    return out


def _std_summary(items: list[dict], first_k: int | None = None) -> dict[str, float]:
    """Mean of the ``std_errors`` of queries or episodes (RMSE from the mean squared error)."""
    if not items or "first_action_mae_std" not in items[0]:
        return {}

    def mse(name):
        return [x[f"{name}_mse_std"] if f"{name}_mse_std" in x else x[f"{name}_rmse_std"] ** 2 for x in items]

    out = {
        "action_rmse_std": float(np.sqrt(np.mean(mse("action")))),
        "action_mae_std": float(np.mean([x["action_mae_std"] for x in items])),
        "first_action_mae_std": float(np.mean([x["first_action_mae_std"] for x in items])),
    }
    if first_k:
        out["first_k"] = first_k
        out["first_k_rmse_std"] = float(np.sqrt(np.mean(mse("first_k"))))
        out["first_k_mae_std"] = float(np.mean([x["first_k_mae_std"] for x in items]))
    return out


def _action_std(repo_id: str, key: str) -> np.ndarray | None:
    """Per-dimension standard deviation of ``key`` over all episodes of a LeRobot dataset."""
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    std = ((lerobot_dataset.LeRobotDatasetMetadata(repo_id).stats or {}).get(key) or {}).get("std")
    return None if std is None else np.maximum(np.asarray(std, dtype=np.float64), 1e-8)


def _to_uint8_hwc(frames) -> np.ndarray:
    """(T, C, H, W) float in [0, 1] or uint8 -> (T, H, W, C) uint8."""
    x = np.asarray(frames)
    if np.issubdtype(x.dtype, np.floating):
        x = (np.clip(x, 0.0, 1.0) * 255).round().astype(np.uint8)
    if x.ndim == 4 and x.shape[1] in (1, 3):
        x = x.transpose(0, 2, 3, 1)
    return x


def _episode_dataset(train_config, data_config, repo_id: str, episode: int):
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    from temporal_ratio.data import video_decode

    video_decode.install()

    from temporal_ratio.data import transforms as _transforms

    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    horizon = train_config.model.action_horizon
    obs_horizon = train_config.model.observation_horizon
    delta = {k: [t / meta.fps for t in range(horizon)] for k in data_config.action_sequence_keys}
    for k in data_config.obs_sequence_keys:
        delta[k] = [t * data_config.frame_skip / meta.fps for t in range(obs_horizon)]
    ds = lerobot_dataset.LeRobotDataset(repo_id, delta_timestamps=delta, episodes=[episode])
    _index_by_episode(ds)
    prompt = _transforms.PromptFromLeRobotTask(meta.tasks) if data_config.prompt_from_task else None
    return ds, prompt


def _index_by_episode(ds) -> None:
    """Re-key ``episode_data_index`` by global episode index.

    LeRobot 0.1.0 builds it for the selected episodes only (positions 0..n-1) but looks it up with
    the global ``episode_index`` of a frame, which breaks any selection other than ``[0, 1, ...]``.
    """
    import torch

    episodes = list(ds.episodes or [])
    if not episodes or episodes == list(range(len(episodes))):
        return
    for key in ("from", "to"):
        old = ds.episode_data_index[key]
        new = torch.zeros(max(episodes) + 1, dtype=old.dtype)
        for pos, ep in enumerate(episodes):
            new[ep] = old[pos]
        ds.episode_data_index[key] = new


def run(
    exp: InferenceExperiment,
    *,
    out_dir: str | pathlib.Path | None = None,
    datasets: list[str] | None = None,
    episodes: list[int] | None = None,
    max_steps: int | None = None,
    prompt: str | None = None,
) -> dict[str, Any]:
    """Evaluate ``exp`` open-loop; returns the report written to ``<out_dir>/report.json``.

    Args (default: ``exp.offline``):
        out_dir: output directory.
        datasets: LeRobot dataset ids (default: the policy config's primary dataset).
        episodes: episode indices, evaluated in every dataset (indices beyond a dataset are skipped).
        max_steps: maximum number of policy queries per episode.
        prompt: override the recorded task prompt.
    """
    import torch

    from temporal_ratio.config import registry
    from temporal_ratio.inference import infer_kwargs, load_policy

    train_config = registry.get_config(exp.policy_config)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    datasets = list(datasets or exp.offline.datasets or [data_config.repo_id])
    episodes = list(episodes or exp.offline.episodes)
    max_steps = max_steps if max_steps is not None else exp.offline.max_steps
    out_dir = pathlib.Path(out_dir or exp.offline.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(exp.seed)
    rng = np.random.default_rng(exp.seed)

    t0 = time.perf_counter()
    policy = load_policy(exp)
    load_s = time.perf_counter() - t0
    kwargs = infer_kwargs(exp, policy)

    report: dict[str, Any] = {"config": as_dict(exp), "load_seconds": round(load_s, 1), "datasets": {}}
    for repo_id in datasets:
        ds_dir = out_dir / repo_id.replace("/", "__")
        ds_dir.mkdir(parents=True, exist_ok=True)
        std = _action_std(repo_id, data_config.action_sequence_keys[0])
        results = {}
        for episode in episodes:
            result = _evaluate_episode(
                policy, kwargs, exp, train_config, data_config, repo_id, episode, max_steps, prompt, rng, ds_dir, std
            )
            if result is not None:
                results[str(episode)] = result
        summaries = [r["summary"] for r in results.values()]
        report["datasets"][repo_id] = {
            "episodes": results,
            "summary": {
                "num_episodes": len(summaries),
                "num_queries": int(sum(x["num_queries"] for x in summaries)),
                "action_mse": float(np.mean([x["action_mse"] for x in summaries])) if summaries else None,
                "action_l1": float(np.mean([x["action_l1"] for x in summaries])) if summaries else None,
                **_std_summary(summaries, exp.offline.first_k),
                "all_finite": all(x["all_finite"] for x in summaries),
                "latency_s_warm_mean": float(np.mean([x["latency_s_warm_mean"] for x in summaries])) if summaries else None,
            },
            "action_std": None if std is None else std.tolist(),
        }
        log.info(f"{repo_id}: " + json.dumps(report["datasets"][repo_id]["summary"]))
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def _evaluate_episode(policy, kwargs, exp, train_config, data_config, repo_id, episode, max_steps, prompt, rng, out_dir,
                      std=None):
    import imageio.v2 as imageio
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
    import torch

    num_episodes = lerobot_dataset.LeRobotDatasetMetadata(repo_id).total_episodes
    if episode >= num_episodes:
        log.warning(f"{repo_id} has {num_episodes} episodes; skipping episode {episode}")
        return None
    if hasattr(policy, "reset_episode"):
        policy.reset_episode()
    ds, prompt_tf = _episode_dataset(train_config, data_config, repo_id, episode)
    action_shape = (train_config.model.action_horizon, train_config.model.action_dim)
    steps = list(range(0, len(ds), exp.replan_every))[: max_steps or None]
    records, pred_all, gt_all = [], [], []
    for t in steps:
        sample = ds[t]
        if prompt_tf is not None:
            sample = prompt_tf(sample)
        future = {k: sample[k] for k in data_config.obs_sequence_keys}
        sample = {k: (v[0] if k in data_config.obs_sequence_keys else v) for k, v in sample.items()}
        for tf in data_config.repack_transforms.inputs:
            sample = tf(sample)
        gt = np.asarray(sample.pop("actions"))
        obs = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in sample.items()}
        if prompt is not None:
            obs["prompt"] = prompt

        noise = rng.standard_normal(action_shape).astype(np.float32)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            out = policy.infer(obs, noise=noise, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - start

        pred = np.asarray(out["actions"])[:, : gt.shape[-1]]
        err = pred - gt[: len(pred)]
        rec = {
            "frame": t,
            "latency_s": round(latency, 3),
            "action_mse": float(np.mean(err**2)),
            "action_l1": float(np.mean(np.abs(err))),
            "first_action_l1": float(np.mean(np.abs(err[0]))),
            "finite": bool(np.isfinite(pred).all()),
        }
        if std is not None:
            rec.update(std_errors(pred, gt[: len(pred)], std[: pred.shape[-1]], exp.offline.first_k))
        tr = out.get("temporal_ratio")
        if tr:
            rec["temporal_ratio"] = {str(k): round(float(v), 4) for k, v in tr.items()}
        if (info := out.get("adaptive_cfg_info")) is not None:
            rec["guidance"] = {k: v for k, v in info.items() if isinstance(v, (int, float, str, bool))}
        video = out.get("future_video")
        if video is not None:
            path = out_dir / f"episode_{episode}_step_{t:05d}.mp4"
            real = np.concatenate([_to_uint8_hwc(future[k]) for k in data_config.obs_sequence_keys], axis=2)
            imageio.mimwrite(path, _side_by_side(np.asarray(video), real), fps=10)
            rec["video"] = str(path.relative_to(out_dir.parent))
        records.append(rec)
        pred_all.append(pred)
        gt_all.append(gt[: len(pred)])
        log.info(f"{repo_id} episode {episode} frame {t}: " + json.dumps({k: rec[k] for k in ("latency_s", "action_mse")}))

    np.savez(out_dir / f"episode_{episode}.npz", frames=np.asarray(steps), predicted=np.asarray(pred_all),
             recorded=np.asarray(gt_all))
    warm = [r["latency_s"] for r in records[1:]] or [r["latency_s"] for r in records]
    return {
        "num_frames": len(ds),
        "steps": records,
        "summary": {
            "num_queries": len(records),
            "action_mse": float(np.mean([r["action_mse"] for r in records])),
            "action_l1": float(np.mean([r["action_l1"] for r in records])),
            **_std_summary(records, exp.offline.first_k),
            "all_finite": all(r["finite"] for r in records),
            "latency_s_warm_mean": float(np.mean(warm)),
        },
    }


def _side_by_side(pred: np.ndarray, real: np.ndarray) -> list[np.ndarray]:
    """Stack predicted (top) and recorded (bottom) frames; recorded frames are resized to match."""
    from PIL import Image

    h, w = pred.shape[1:3]
    frames = []
    for i in range(min(len(pred), len(real))):
        r = np.asarray(Image.fromarray(real[i]).resize((w, h), Image.BILINEAR))
        frames.append(np.concatenate([pred[i], r], axis=0))
    return frames
