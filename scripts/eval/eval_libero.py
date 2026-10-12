#!/usr/bin/env python3
"""
Multi-GPU, multi-node parallel LIBERO evaluation via torchrun.

Each GPU gets its own policy instance. Work items (task_suite, task_id) are
distributed round-robin across all workers. Each worker saves videos and
partial results independently.  Rank 0 aggregates per-suite results after
a distributed barrier.

Usage (single node, all GPUs):
    torchrun --standalone --nproc_per_node=auto \
        scripts/libero_eval_complete_parallel.py

Usage (single node, 4 GPUs):
    torchrun --standalone --nproc_per_node=4 \
        scripts/libero_eval_complete_parallel.py

Usage (multi-node, 2 nodes × 4 GPUs, using rendezvous):
    # On every node:
    torchrun --nnodes=2 --nproc_per_node=4 \
        --rdzv_backend=c10d --rdzv_endpoint=<master_ip>:29400 \
        scripts/libero_eval_complete_parallel.py

Usage (single GPU for debugging):
    torchrun --standalone --nproc_per_node=1 \
        scripts/libero_eval_complete_parallel.py
"""

import collections
import dataclasses
import datetime
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import sys
import pathlib
import shutil
from typing import Optional

import imageio
import numpy as np
import random
import torch
import torch.distributed as dist
import tqdm
import tyro
from PIL import Image, ImageDraw, ImageFont


def _ensure_libero_config():
    """Auto-create ~/.libero/config.yaml with defaults so libero never prompts."""
    config_dir = os.path.expanduser("~/.libero")
    config_file = os.path.join(config_dir, "config.yaml")
    if os.path.exists(config_file):
        return
    import importlib
    spec = importlib.util.find_spec("libero.libero")
    if spec is None or spec.origin is None:
        return
    libero_root = os.path.dirname(os.path.abspath(spec.origin))
    import yaml
    paths = {
        "benchmark_root": libero_root,
        "bddl_files": os.path.join(libero_root, "bddl_files"),
        "init_states": os.path.join(libero_root, "init_files"),
        "datasets": os.path.join(os.path.dirname(libero_root), "datasets"),
        "assets": os.path.join(libero_root, "assets"),
    }
    os.makedirs(config_dir, exist_ok=True)
    with open(config_file, "w") as f:
        yaml.dump(paths, f)
    print(f"Auto-created libero config at {config_file} pointing to {libero_root}")


_ensure_libero_config()


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that writes through tqdm.write to avoid corrupting progress bars."""

    def emit(self, record):
        msg = self.format(record)
        tqdm.tqdm.write(msg)


"""
Install robosuite 1.4.0
"""

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 224

# ─── Max steps per suite ────────────────────────────────────────────────────────
MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,
    "libero_spatial_ood": int(220 * 1.5),
    "libero_object": 280,
    "libero_object_ood": int(280 * 1.5),
    "libero_goal": 300,
    "libero_goal_ood": int(300 * 1.5),
    "libero_10": 520,
    "libero_90": 400,
}


# ─── CLI Args ────────────────────────────────────────────────────────────────────
@dataclasses.dataclass
class Args:
    """Arguments for the parallel evaluation script."""

    config: str = "videoVLA_libero_short"

    # Mode of checkpoint
    mode: str = "videoVLA_libero_short_flat"
    # Checkpoint step to load.
    checkpoint_step: str = "30000"

    # Evaluate on out-of-distribution tasks
    eval_ood: bool = False
    eval_90: bool = False
    eval_all: bool = False
    # Explicit suites (overrides the flags above), e.g. --task_suites libero_spatial libero_goal
    task_suites: tuple[str, ...] = ()
    # Only these task ids within each suite (default: all)
    task_ids: tuple[int, ...] = ()

    # ── Model parameters ──────────────────────────────────────────────────────
    resize_size: int = 224
    replan_steps: int = 10
    inference_denoising_steps: int = 2
    vid2world_mode: str = "libero_flat_short"
    vid2world_ckpt: Optional[str] = None  # Override video model checkpoint path (Cosmos or Wan)

    # ── Video generation context ─────────────────────────────────────────────
    num_gen_latent_frames: Optional[int] = None  # Generate this many latent frames (default = 5, same as action).
                                                  # Set to 24 (=93 video frames) for long-context goal planning.
                                                  # Features are always cropped to first 5 for the action head.

    # ── LIBERO env parameters ─────────────────────────────────────────────────
    num_steps_wait: int = 10
    num_trials_per_task: int = 10

    # ── Output ────────────────────────────────────────────────────────────────
    video_out_path: str = "data/libero/videos/"

    seed: int = 7

    # ── Rollout data recording ───────────────────────────────────────────────
    save_rollout_data: bool = False               # Save full rollout data (actions, video predictions, observations)

    # ── Attention visualization ───────────────────────────────────────────────
    visualize_attention: bool = False             # Capture and save attention heatmaps
    attn_layers: Optional[str] = None            # Comma-separated layer indices (e.g. "0,5,11"); None = all
    attn_denoise_step: int = -1                  # Which denoising step to capture (-1 = last)
    attn_head_reduction: str = "mean"            # "mean" or "max" across attention heads
    attn_every_n_replans: int = 1                # Save attention every N replan steps (1 = every replan)

    # ── Wandb logging (optional) ──────────────────────────────────────────────
    use_wandb: bool = False                      # Enable wandb logging (rank 0 only)
    wandb_project: str = "libero-eval"            # Wandb project name
    wandb_entity: Optional[str] = None            # Wandb entity (team/user); None = default
    wandb_run_name: Optional[str] = None          # Wandb run name; None = auto-generated

    # Note: parallelism (rank, world_size, local_rank) is handled by torchrun
    # via environment variables. No extra CLI args needed.


@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str


# ─── Helpers (no CUDA / heavy imports) ───────────────────────────────────────────

def _trial_seed(base_seed: int, suite_name: str, task_id: int, trial_id: int) -> int:
    """Deterministic seed for a specific (suite, task, trial) combination.

    Guarantees the same seed regardless of which GPU runs the trial.
    """
    key = f"{base_seed}:{suite_name}:{task_id}:{trial_id}"
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**31)


def _get_task_suite_list(args: Args) -> list[str]:
    if args.task_suites:
        return list(args.task_suites)
    if args.eval_90:
        return ["libero_90"]
    if args.eval_ood:
        return ["libero_spatial_ood", "libero_object_ood", "libero_goal_ood"]
    if args.eval_all:
        return [
            "libero_spatial", "libero_object", "libero_goal", "libero_10",
            "libero_spatial_ood", "libero_object_ood", "libero_goal_ood",
            "libero_90"
        ]
    return [
        "libero_spatial", "libero_object", "libero_goal", "libero_10" # ID tasks
    ]


def _num_trials(args: Args) -> int:
    return args.num_trials_per_task if args.eval_ood else args.num_trials_per_task


def _video_out_base(args: Args) -> str:
    return (
        f"{args.video_out_path}/{args.config}/{args.mode}"
        f"/step_{args.checkpoint_step}/inf_{args.inference_denoising_steps}"
        f"/vid2world_{args.vid2world_mode}"
    )


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


# ─── Video concatenation (rollout + predicted) ──────────────────────────────────

def save_combined_rollout_video(
    replay_images: list[np.ndarray],
    replay_wrist_images: list[np.ndarray],
    video_plan: list,
    replan_steps: int,
    task_suite_name: str,
    task_id: int,
    episode_idx: int,
    success: bool,
    save_path: str,
    fps: int = 10,
) -> str:
    """Save an MP4 with rollout on top row and predicted future video on bottom.

    Layout per frame:
        ┌────────────────┐┌────────────────┐
        │  rollout frame ││  rollout wrist │
        ├────────────────┤├────────────────┤
        │ predicted frame││ predicted wrist|
        └────────────────┘└────────────────┘

    The predicted frames are expanded from the collected video_plan:
    each element in video_plan is an array of shape (T, H, W, C) predicted
    at a replan step.  We tile them to align with the rollout timeline.
    """
    if not replay_images or not video_plan:
        return ""

    # Build aligned predicted-frame timeline
    # video_plan[i] was generated at rollout step i * replan_steps
    # and covers the next len(video_plan[i]) predicted frames.
    n_rollout = len(replay_images)
    target_h, target_w = replay_images[0].shape[:2]

    predicted_images = []
    predicted_wrist_images = []

    for sequence in video_plan:
        T, H, W, C = sequence.shape
        for i in range(T):
            predicted_images.append(sequence[i, :H, :W//2, :])
            predicted_wrist_images.append(sequence[i, :H, W//2:, :])

    predicted_images = predicted_images[:n_rollout]
    predicted_wrist_images = predicted_wrist_images[:n_rollout]

    # Build text header
    text_height = 50
    font_size = 16
    try:
        font = ImageFont.truetype("DejaVuSans", font_size)
    except IOError:
        font = ImageFont.load_default()

    frames = []
    for i in range(n_rollout):
        rollout_frame = replay_images[i]
        rollout_wrist_frame = replay_wrist_images[i]
        predicted_frame = predicted_images[i]
        predicted_wrist_frame = predicted_wrist_images[i]
        
        # Ensure both are uint8
        if rollout_frame.dtype != np.uint8:
            rollout_frame = np.clip(rollout_frame, 0, 255).astype(np.uint8)
        if rollout_wrist_frame.dtype != np.uint8:
            rollout_wrist_frame = np.clip(rollout_wrist_frame, 0, 255).astype(np.uint8)
        if predicted_frame.dtype != np.uint8:
            predicted_frame = np.clip(predicted_frame, 0, 255).astype(np.uint8)
        if predicted_wrist_frame.dtype != np.uint8:
            predicted_wrist_frame = np.clip(predicted_wrist_frame, 0, 255).astype(np.uint8)

        combined_rollout_frame = np.hstack([rollout_frame, rollout_wrist_frame])
        combined_predicted_frame = np.hstack([predicted_frame, predicted_wrist_frame])

        # Text area
        text_area = np.ones((text_height, 2*target_w, 3), dtype=np.uint8) * 255
        text_img = Image.fromarray(text_area)
        draw = ImageDraw.Draw(text_img)
        header = f"{task_suite_name} | task {task_id} | ep {episode_idx} | t={i}"
        draw.text((4, 2), header, font=font, fill=(0, 0, 0))
        draw.text((4, 22), "top: rollout  |  bottom: predicted", font=font, fill=(80, 80, 80))
        text_area = np.array(text_img)

        combined = np.vstack([text_area, combined_rollout_frame, combined_predicted_frame])
        frames.append(combined)

    imageio.mimwrite(
        str(save_path), frames, fps=fps,
        codec="libx264", pixelformat="yuv420p", macro_block_size=1,
        output_params=["-crf", "23", "-preset", "medium"],
    )
    logging.info(f"Saved combined rollout+predicted video: {save_path}")
    return str(save_path)


def _env_worker(conn, bddl_file: str, resolution: int, gpu_id: int):
    """Child process: create a LIBERO env and serve commands over *conn*.

    Runs in a *spawn*-ed process so the GL state is clean (no transformers
    or PyOpenGL pollution from the parent).
    """
    try:
        from libero.libero.envs import OffScreenRenderEnv

        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=resolution,
            camera_widths=resolution,
            render_gpu_device_id=gpu_id,
        )
        conn.send(("ok", None))

        while True:
            cmd, payload = conn.recv()
            if cmd == "seed":
                env.seed(payload)
                conn.send(("ok", None))
            elif cmd == "reset":
                env.reset()
                conn.send(("ok", None))
            elif cmd == "get_observation":
                if hasattr(env, "get_observation"):  # LIBERO fork with the OOD suites
                    obs = env.get_observation()
                else:  # upstream LIBERO: query the wrapped robosuite environment
                    obs = env.env._get_observations(force_update=True)
                conn.send(("ok", obs))
            elif cmd == "step":
                obs, reward, done, info = env.step(payload)
                conn.send(("ok", (obs, reward, done, info)))
            elif cmd == "close":
                env.close()
                conn.send(("ok", None))
                break
            else:
                conn.send(("error", f"unknown command: {cmd}"))
    except Exception as e:
        conn.send(("error", str(e)))
    finally:
        conn.close()


class SubprocessEnv:
    """Drop-in replacement for OffScreenRenderEnv that runs in a child process."""

    def __init__(self, bddl_file: str, resolution: int, gpu_id: int):
        ctx = mp.get_context("spawn")
        self._parent_conn, child_conn = ctx.Pipe()
        self._proc = ctx.Process(
            target=_env_worker,
            args=(child_conn, str(bddl_file), resolution, gpu_id),
            daemon=True,
        )
        self._proc.start()
        child_conn.close()  # parent doesn't use the child end
        status, _ = self._parent_conn.recv()
        if status != "ok":
            raise RuntimeError(f"env worker failed to start: {_}")

    def _call(self, cmd, payload=None):
        self._parent_conn.send((cmd, payload))
        status, result = self._parent_conn.recv()
        if status != "ok":
            raise RuntimeError(f"env worker error on {cmd!r}: {result}")
        return result

    def seed(self, s):
        self._call("seed", s)

    def reset(self):
        self._call("reset")

    def get_observation(self):
        return self._call("get_observation")

    def step(self, action):
        return self._call("step", action)

    def close(self):
        try:
            self._call("close")
        except (BrokenPipeError, EOFError, OSError):
            pass
        self._proc.join(timeout=5)
        if self._proc.is_alive():
            self._proc.kill()


def _get_libero_env(task, resolution, seed, render_gpu_device_id=0):
    from libero.libero import get_libero_path

    task_description = task.language
    task_bddl_file = str(
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )

    env = SubprocessEnv(task_bddl_file, resolution, render_gpu_device_id)
    env.seed(seed)
    return env, task_description


# ─── Build work items (lightweight, no CUDA) ─────────────────────────────────────

def _build_work_items(args: Args) -> list[tuple[str, int, int]]:
    """Return a list of (task_suite_name, task_id, trial_id) triples to evaluate."""
    from libero.libero import benchmark as libero_benchmark

    task_suite_list = _get_task_suite_list(args)
    benchmark_dict = libero_benchmark.get_benchmark_dict()
    num_trials = _num_trials(args)
    work_items: list[tuple[str, int, int]] = []
    for suite_name in task_suite_list:
        task_suite = benchmark_dict[suite_name]()
        task_ids = [t for t in range(task_suite.n_tasks) if not args.task_ids or t in args.task_ids]
        for task_id in task_ids:
            for trial_id in range(num_trials):
                work_items.append((suite_name, task_id, trial_id))
    return work_items


# ─── Single-episode rollout ─────────────────────────────────────────────────────

def _run_episode(
    args: Args,
    policy,
    env,
    task_description: str,
    task_suite_name: str,
    task_id: int,
    episode_idx: int,
    max_steps: int,
    video_out_path: str,
) -> dict:
    """
    Run a single episode.  Returns dict with keys:
        success (bool), skipped (bool), skip_reason (str|None)
    Videos / future-video arrays are saved to disk inside this function.
    """
    from openpi_client import image_tools

    # Check if this episode was already completed
    check_path = pathlib.Path(video_out_path) / f"task_{task_id}" / f"*_{episode_idx}.mp4"
    searched_files = list(check_path.parent.glob(check_path.name))
    if searched_files:
        is_success = "success" in searched_files[0].name
        if is_success:
            env.reset()
            _ = env.get_observation()
        return {"success": is_success, "skipped": True, "skip_reason": "video_exists"}

    save_future_video_path = (
        pathlib.Path(video_out_path) / f"task_{task_id}" / f"future_video_{episode_idx}.npy"
    )
    if save_future_video_path.exists():
        return {"success": False, "skipped": True, "skip_reason": "future_video_exists"}

    # Reset environment
    env.reset()
    obs = env.get_observation()

    action_plan: collections.deque = collections.deque()
    video_plan: collections.deque = collections.deque()
    replay_images: list = []
    replay_wrist_images: list = []
    future_video = None
    replan_count = 0

    # Rollout data recording
    rollout_data_actions_executed: list = []       # per-step executed action
    rollout_data_action_chunks: list = []          # full action prediction per replan
    rollout_data_video_predictions: list = []      # full video prediction per replan
    rollout_data_replan_steps: list = []            # rollout step index of each replan
    rollout_data_features: list = []               # layer-19 features per replan (T*H*W, D)
    rollout_data_states: list = []                 # per-step observation state (eef_pos + axisangle + gripper)
    rollout_data_wrist_images: list = []           # per-replan wrist image (for offline action head replay)

    # Parse attention layer indices if provided
    attn_layers = None
    if args.visualize_attention and args.attn_layers is not None:
        attn_layers = [int(x) for x in args.attn_layers.split(",")]

    t = 0
    done = False
    skip_episode = False

    pbar = tqdm.tqdm(
        total=max_steps + args.num_steps_wait,
        desc=f"{task_suite_name} task {task_id} ep {episode_idx}",
        leave=False, position=1,
    )

    while t < max_steps + args.num_steps_wait:
        try:
            # Wait for objects to stabilize in sim
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                pbar.update(1)
                continue

            # Preprocess images (rotate 180 degrees to match training)
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
            )
            replay_images.append(img)
            replay_wrist_images.append(wrist_img)

            # Record observation state for offline replay
            if args.save_rollout_data:
                obs_state = np.concatenate((
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ))
                rollout_data_states.append(obs_state)

            if not action_plan:
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    ),
                    "prompt": str(task_description),
                }

                # Attention visualization on selected replan steps
                use_attn = (
                    args.visualize_attention
                    and hasattr(policy, "infer_with_attention")
                    and replan_count % args.attn_every_n_replans == 0
                )
                if use_attn:
                    attn_dir = pathlib.Path(video_out_path) / f"task_{task_id}" / "attention"
                    attn_dir.mkdir(parents=True, exist_ok=True)
                    attn_save = str(attn_dir / f"ep{episode_idx}_replan{replan_count}.png")
                    policy_output = policy.infer_with_attention(
                        element,
                        save_path=attn_save,
                        denoise_step_idx=args.attn_denoise_step,
                        layers=attn_layers,
                        head_reduction=args.attn_head_reduction,
                    )
                    # Also save the raw segment_attention array
                    attn_info = policy_output.get("attention_info")
                    if attn_info is not None:
                        np.save(
                            str(attn_dir / f"ep{episode_idx}_replan{replan_count}_seg_attn.npy"),
                            attn_info["segment_attention"],
                        )
                else:
                    infer_kwargs = {}
                    if args.num_gen_latent_frames is not None:
                        infer_kwargs["num_gen_latent_frames"] = args.num_gen_latent_frames
                    policy_output = policy.infer(element, **infer_kwargs)

                replan_count += 1
                action_chunk = policy_output["actions"]
                future_video = policy_output.get("future_video", None)
                assert len(action_chunk) >= args.replan_steps, (
                    f"Replan every {args.replan_steps} steps but policy predicts {len(action_chunk)}."
                )
                action_plan.extend(action_chunk[: args.replan_steps])
                if future_video is not None:
                    video_plan.append(future_video[: args.replan_steps])

                # Record rollout data
                if args.save_rollout_data:
                    rollout_data_replan_steps.append(t - args.num_steps_wait)
                    rollout_data_action_chunks.append(np.array(action_chunk))
                    if future_video is not None:
                        rollout_data_video_predictions.append(np.array(future_video))
                    features = policy_output.get("features", None)
                    if features is not None:
                        rollout_data_features.append(np.array(features))
                    # Save wrist image at replan for offline replay
                    rollout_data_wrist_images.append(wrist_img.copy())

            action = action_plan.popleft()
            if args.save_rollout_data:
                rollout_data_actions_executed.append(np.array(action))
            obs, reward, done, info = env.step(action.tolist())
            if done:
                break
            t += 1
            pbar.update(1)

        except Exception as e:
            logging.error(
                f"Exception during rollout suite={task_suite_name} task={task_id} ep={episode_idx}: {e}",
                exc_info=True,
            )
            skip_episode = True
            break

    pbar.close()

    if skip_episode:
        return {"success": False, "skipped": True, "skip_reason": "exception"}

    # ── Save outputs ─────────────────────────────────────────────────────────
    suffix = "success" if done else "failure"
    task_segment = f"task_{task_id}"

    save_path = (
        pathlib.Path(video_out_path) / task_segment / f"rollout_{suffix}_{episode_idx}.mp4"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # imageio.mimwrite(save_path, [np.asarray(x) for x in replay_images], fps=10)

    # Save combined rollout + predicted video
    if video_plan:
        save_combined_rollout_video(
            replay_images=replay_images,
            replay_wrist_images=replay_wrist_images,
            video_plan=video_plan,
            replan_steps=args.replan_steps,
            task_suite_name=task_suite_name,
            task_id=task_id,
            episode_idx=episode_idx,
            success=done,
            save_path=save_path,
        )

    prompt_path = pathlib.Path(video_out_path) / task_segment / "prompt.txt"
    if not prompt_path.exists():
        with open(prompt_path, "w") as f:
            f.write(str(task_description))

    # Save rollout data
    if args.save_rollout_data:
        rollout_data_dir = pathlib.Path(video_out_path) / task_segment / "rollout_data"
        rollout_data_dir.mkdir(parents=True, exist_ok=True)
        ep_file = rollout_data_dir / f"ep_{episode_idx}.npz"
        save_dict = {
            "inference_denoising_steps": np.array(args.inference_denoising_steps),
            "replan_steps": np.array(args.replan_steps),
            "success": np.array(done),
            "actions_executed": np.array(rollout_data_actions_executed) if rollout_data_actions_executed else np.array([]),
            "replan_step_indices": np.array(rollout_data_replan_steps),
            "replay_images": np.array(replay_images),
            "observation_states": np.array(rollout_data_states) if rollout_data_states else np.array([]),
        }
        # Save full action predictions per replan (variable length, use object array)
        for i, chunk in enumerate(rollout_data_action_chunks):
            save_dict[f"action_chunk_{i:03d}"] = chunk
        # Save full video predictions per replan
        for i, vid in enumerate(rollout_data_video_predictions):
            save_dict[f"video_prediction_{i:03d}"] = vid
        # Save layer features per replan (for replan coherence analysis)
        for i, feat in enumerate(rollout_data_features):
            save_dict[f"features_{i:03d}"] = feat
        # Save wrist images at each replan (for offline action head replay)
        for i, wimg in enumerate(rollout_data_wrist_images):
            save_dict[f"wrist_image_{i:03d}"] = wimg
        np.savez_compressed(str(ep_file), **save_dict)
        logging.info(f"Saved rollout data: {ep_file}")

    return {"success": bool(done), "skipped": False, "skip_reason": None}


# ─── Per-task runner (all episodes for one task, sequential for seed compat) ───

def _run_task(
    args: Args,
    policy,
    task_suite_name: str,
    task_id: int,
    trial_ids: list[int],
) -> dict:
    """
    Run specific trial_ids for a single (suite, task_id).
    Each trial is seeded deterministically via _trial_seed so the result
    is identical regardless of which GPU runs it.
    Returns dict with per-episode results.
    """
    from libero.libero import benchmark as libero_benchmark

    benchmark_dict = libero_benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)

    max_steps = MAX_STEPS_BY_SUITE.get(task_suite_name)
    if max_steps is None:
        raise ValueError(f"Unknown task suite: {task_suite_name}")

    video_out_path = f"{_video_out_base(args)}/{task_suite_name}"
    pathlib.Path(video_out_path).mkdir(parents=True, exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # Create env once; we reseed before each trial
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed, render_gpu_device_id=local_rank)

    task_results = {"suite": task_suite_name, "task_id": task_id, "episodes": []}
    for episode_idx in trial_ids:
        # Deterministic per-trial seed
        seed = _trial_seed(args.seed, task_suite_name, task_id, episode_idx)
        env.seed(seed)
        np.random.seed(seed)

        ep_result = _run_episode(
            args, policy, env, task_description,
            task_suite_name, task_id, episode_idx,
            max_steps, video_out_path,
        )
        ep_result["episode_idx"] = episode_idx
        task_results["episodes"].append(ep_result)
        logging.info(
            f"[{task_suite_name}] task={task_id} ep={episode_idx} seed={seed} "
            f"success={ep_result['success']} skipped={ep_result['skipped']}"
        )

    env.close()
    return task_results





def _queue_dir(args: Args) -> pathlib.Path:
    return pathlib.Path(_video_out_base(args)) / "_worker_results" / "_queue"


def _try_claim(queue_dir: pathlib.Path, suite_name: str, task_id: int, trial_id: int, rank: int) -> bool:
    """Atomically claim a work item. Uses O_CREAT|O_EXCL which is NFS-safe."""
    claim_path = queue_dir / f"{suite_name}__task{task_id:04d}__trial{trial_id:04d}.claimed"
    try:
        fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(rank).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def _write_worker_done(args: Args, global_rank: int, results: list[dict]):
    """Write per-worker results JSON so the aggregator knows this worker is done."""
    out_dir = pathlib.Path(_video_out_base(args)) / "_worker_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"worker_{global_rank}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


# ─── Aggregation (runs on rank 0 after barrier) ─────────────────────────────────

def _aggregate_results(args: Args, world_size: int):
    """
    Load all worker result files and aggregate per-suite results
    into results.txt for each suite + a combined results_all.json.
    """
    worker_dir = pathlib.Path(_video_out_base(args)) / "_worker_results"
    expected_files = [worker_dir / f"worker_{r}.json" for r in range(world_size)]

    # ── Load and merge ────────────────────────────────────────────────────────
    suite_stats: dict[str, dict[str, int]] = {}
    for wf in expected_files:
        if not wf.exists():
            logging.warning(f"Missing worker result file: {wf}")
            continue
        with open(wf) as f:
            worker_results = json.load(f)
        for task_result in worker_results:
            suite = task_result["suite"]
            if suite not in suite_stats:
                suite_stats[suite] = {"total_episodes": 0, "total_successes": 0}
            for ep in task_result["episodes"]:
                suite_stats[suite]["total_episodes"] += 1
                if ep["success"]:
                    suite_stats[suite]["total_successes"] += 1

    # ── Write final per-suite results ─────────────────────────────────────────
    for suite, stats in suite_stats.items():
        ep = stats["total_episodes"]
        succ = stats["total_successes"]
        rate = succ / ep if ep > 0 else 0.0
        results_file = pathlib.Path(_video_out_base(args)) / suite / "results.txt"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, "w") as f:
            f.write(f"Total success rate: {rate}\n")
            f.write(f"Total episodes: {ep}\n")
        logging.info(f"Suite {suite}: {succ}/{ep} = {rate*100:.1f}%")

    # ── Write combined summary ────────────────────────────────────────────────
    summary_path = pathlib.Path(_video_out_base(args)) / "results_all.json"
    with open(summary_path, "w") as f:
        json.dump(suite_stats, f, indent=2)
    logging.info(f"Aggregated results written to {summary_path}")

    # ── Wandb logging ─────────────────────────────────────────────────────────
    if args.use_wandb:
        try:
            import wandb

            run_name = args.wandb_run_name or f"{args.config}--{args.mode}--step{args.checkpoint_step}"
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=dataclasses.asdict(args),
            )

            # Log per-suite metrics
            overall_episodes = 0
            overall_successes = 0
            for suite, stats in suite_stats.items():
                ep = stats["total_episodes"]
                succ = stats["total_successes"]
                rate = succ / ep if ep > 0 else 0.0
                wandb.log({
                    f"per_suite/{suite}/success_rate": rate,
                    f"per_suite/{suite}/total_episodes": ep,
                    f"per_suite/{suite}/total_successes": succ,
                })
                overall_episodes += ep
                overall_successes += succ

            # Log overall metrics
            overall_rate = overall_successes / overall_episodes if overall_episodes > 0 else 0.0
            wandb.log({
                "overall/success_rate": overall_rate,
                "overall/total_episodes": overall_episodes,
                "overall/total_successes": overall_successes,
            })

            # Log a summary table
            table = wandb.Table(columns=["suite", "success_rate", "successes", "episodes"])
            for suite in sorted(suite_stats.keys()):
                s = suite_stats[suite]
                ep = s["total_episodes"]
                succ = s["total_successes"]
                table.add_data(suite, succ / ep if ep > 0 else 0.0, succ, ep)
            table.add_data("OVERALL", overall_rate, overall_successes, overall_episodes)
            wandb.log({"results_table": table})

            wandb.finish()
            logging.info("Wandb logging complete.")
        except ImportError:
            logging.warning("wandb not installed — skipping wandb logging. Install with: pip install wandb")
        except Exception as e:
            logging.warning(f"Wandb logging failed: {e}")


# ─── Main ────────────────────────────────────────────────────────────────────────

class DummyPolicy:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def infer(self, element):
        return {
            "actions": np.zeros((17, 7)),
            "future_video": np.zeros((17, 448, 448, 3))
        }

def get_policy(args, device, dummy=False):
    """Action head + Cosmos video model of ``$CHECKPOINT_DIR`` (a checkpoint step directory)."""
    from temporal_ratio.config import registry as _config
    from temporal_ratio.inference import video_checkpoint
    from temporal_ratio.policies import factory as _policy_config
    from temporal_ratio.policies import video_policy as _policy

    if dummy:
        return DummyPolicy()

    # --mode <config>[_feature_<block>][_diffsteps_<N>]: DiT block read by the action head and the
    # length of the video denoising schedule.
    num_global_denoising_steps = 35
    if "diffsteps" in args.mode:
        num_global_denoising_steps = int(args.mode.split("_diffsteps_")[1])
    mode = args.mode.split("_diffsteps_")[0]
    intermediate_feature_id = int(mode.split("_feature_")[1]) if "_feature_" in mode else 19

    checkpoint_dir = os.environ.get("CHECKPOINT_DIR")
    if not checkpoint_dir:
        raise SystemExit("set CHECKPOINT_DIR to a checkpoint step directory (model.safetensors, assets/, ...)")
    policy_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(
        policy_config, checkpoint_dir, pytorch_device=device, norm_stats=os.environ.get("NORM_STATS_PATH")
    )
    vid2world_ckpt = args.vid2world_ckpt or str(video_checkpoint(pathlib.Path(checkpoint_dir), "auto"))

    return _policy.VideoModelPolicyWrapper(
        policy,
        device=device,
        inference_denoising_steps=args.inference_denoising_steps,
        vid2world_mode=policy_config.video_setup,
        is_causal=False,
        feature_id=intermediate_feature_id,
        num_global_denoising_steps=num_global_denoising_steps,
        vid2world_ckpt=vid2world_ckpt,
    )


def main(args: Args) -> None:
    
    # ── Read torchrun environment variables ───────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)

    # ── Init process group for barrier synchronisation ────────────────────────
    if world_size > 1:
        dist.init_process_group(backend="gloo", timeout=datetime.timedelta(hours=4))

    log_fmt = f"[rank {global_rank}/{world_size} | {device}] %(levelname)s %(message)s"
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(log_fmt))
    root_logger.addHandler(handler)
    logging.info(f"Worker started — local_rank={local_rank}, device={device}")

    # ── Build global work list (suite, task_id, trial_id) ───────────────────
    work_items = _build_work_items(args)
    logging.info(f"Total work items (suite, task_id, trial_id): {len(work_items)}")

    # ── Skip individual trials that already have results ──────────────────────
    remaining_work_items = []
    for suite_name, task_id, trial_id in work_items:
        results_dir = pathlib.Path(_video_out_base(args)) / suite_name / f"task_{task_id}"
        trial_videos = list(results_dir.glob(f"*_{trial_id}.mp4"))
        if trial_videos:
            pass  # already done
        else:
            remaining_work_items.append((suite_name, task_id, trial_id))
    work_items = remaining_work_items
    logging.info(f"Work items after skipping completed trials: {len(remaining_work_items)}")

    if not work_items:
        logging.info("All trials already evaluated. Nothing to do.")
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        return

    # ── Set up dynamic work queue (NFS-safe atomic claiming) ───────────────
    queue_dir = _queue_dir(args)
    if global_rank == 0:
        if queue_dir.exists():
            shutil.rmtree(queue_dir)
        queue_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    # ── Create policy on this GPU ─────────────────────────────────────────
    policy = get_policy(args, device)
    logging.info(f"Policy loaded on {device}")

    # ── Dynamic work loop ─────────────────────────────────────────────────
    # Group items by (suite, task_id) and shuffle groups per-rank so
    # workers don't all contend on the same tasks. Within each group we
    # claim individual trials, keeping env reuse across trials.
    groups: dict[tuple[str, int], list[int]] = {}
    for suite_name, task_id, trial_id in work_items:
        groups.setdefault((suite_name, task_id), []).append(trial_id)

    group_keys = list(groups.keys())
    random.Random(global_rank).shuffle(group_keys)

    all_results: list[dict] = []
    claimed_count = 0
    successes = 0
    overall_bar = tqdm.tqdm(
        total=len(work_items), desc=f"[rank {global_rank}] Trials", unit="trial",
        position=0, dynamic_ncols=True,
    )
    for suite_name, task_id in group_keys:
        my_trials = []
        for trial_id in groups[(suite_name, task_id)]:
            if _try_claim(queue_dir, suite_name, task_id, trial_id, global_rank):
                my_trials.append(trial_id)
                claimed_count += 1
        if not my_trials:
            continue
        overall_bar.set_postfix(suite=suite_name, task=task_id, ok=f"{successes}/{claimed_count}")
        try:
            task_result = _run_task(args, policy, suite_name, task_id, my_trials)
        except Exception as e:
            logging.error(
                f"Task {suite_name} task_id={task_id} crashed: {e}",
                exc_info=True,
            )
            task_result = {
                "suite": suite_name,
                "task_id": task_id,
                "episodes": [],
                "error": str(e),
            }
        all_results.append(task_result)
        for ep in task_result.get("episodes", []):
            if ep.get("success"):
                successes += 1
        overall_bar.update(len(my_trials))
        overall_bar.set_postfix(suite=suite_name, task=task_id, ok=f"{successes}/{claimed_count}")

    overall_bar.close()
    logging.info(f"Completed {claimed_count} trials, {successes} successes (dynamic queue)")

    # ── Write partial results ─────────────────────────────────────────────────
    _write_worker_done(args, global_rank, all_results)
    logging.info("Worker finished — waiting at barrier for all ranks …")

    # ── Barrier: wait for every rank to finish ────────────────────────────────
    if world_size > 1:
        dist.barrier()

    # ── Aggregate on rank 0 ──────────────────────────────────────────────────
    if global_rank == 0:
        _aggregate_results(args, world_size)

    if world_size > 1:
        dist.destroy_process_group()

    crashed = [f"{r['suite']}/{r['task_id']}" for r in all_results if r.get("error")]
    if crashed:
        logging.error(f"{len(crashed)} task(s) crashed on rank {global_rank}: {crashed}")
    return 1 if crashed else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[main] %(levelname)s %(message)s")
    args = tyro.cli(Args)
    sys.exit(main(args))