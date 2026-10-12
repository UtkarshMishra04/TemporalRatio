#!/usr/bin/env python3
"""Robust TR probe: per-layer, per-action, per-denoise-step attention capture
on a fixed set of real LIBERO observations.

For each checkpoint we:
  1. Load action head + VideoProcessor.
  2. For each task (sampled via LeRobot parquets) gather a few frames.
  3. For each frame:
      a. Run V2W (cond) and V2W (uncond) once each — cache features.
      b. Run visualize_attention at denoise_step_idx ∈ {0, num_steps//2, num_steps-1}
         capturing ALL Gemma layers.
      c. Save per (layer, denoise_step, action_idx): attention mass on each segment.
  4. Dump to disk. Downstream analysis computes TR variants.

Usage:
  CUDA_VISIBLE_DEVICES=0 python final_analysis_scripts/27_tr_probe.py \
      --ckpt checkpoints/.../15500 \
      --ckpt_name v3_joint_r16 \
      --task_ids 0 1 2 3 4 \
      --frames_per_task 6 \
      --out results/tr_probe/v3_joint_r16
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import pyarrow.parquet as pq
import torch
from io import BytesIO
from PIL import Image

from temporal_ratio.policies import factory as _policy_config
from temporal_ratio.policies import video_policy as _policy
from temporal_ratio.config import registry as _config
from temporal_ratio.models import base as _model
from temporal_ratio.models.video_model import IMAGE_SIZES


LEROBOT_ROOT = Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot")) / "physical-intelligence/libero"
NUM_LATENT_FRAMES = 5
NUM_VIDEO_FRAMES = 4 * (NUM_LATENT_FRAMES - 1) + 1  # 17


def load_task_frames(task_index, n_frames, frame_stride=20):
    """Sample frames from a given task_index using the LeRobot dataset."""
    with open(LEROBOT_ROOT / "meta/tasks.jsonl") as f:
        tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in f}
    prompt = tasks[task_index]

    # Iterate parquet files until we find episodes for this task, collect frames
    chosen = []
    chunk_dirs = sorted([d for d in (LEROBOT_ROOT / "data").iterdir() if d.is_dir()])
    for chunk_dir in chunk_dirs:
        if len(chosen) >= n_frames:
            break
        for p in sorted(chunk_dir.glob("*.parquet")):
            if len(chosen) >= n_frames:
                break
            t = pq.read_table(str(p))
            task_idx_col = t.column("task_index").to_numpy()
            mask = task_idx_col == task_index
            if not mask.any():
                continue
            inds = np.where(mask)[0]
            # Pick frames spaced out
            picks = inds[::frame_stride][: n_frames - len(chosen)]
            for idx in picks:
                row = t.slice(int(idx), 1).to_pydict()
                img = Image.open(BytesIO(row["image"][0]["bytes"])).convert("RGB").resize((224, 224))
                wrist = Image.open(BytesIO(row["wrist_image"][0]["bytes"])).convert("RGB").resize((224, 224))
                # Combined image (front | wrist)
                combined = Image.new("RGB", (448, 224))
                combined.paste(img, (0, 0))
                combined.paste(wrist, (224, 0))
                state = np.asarray(row["state"][0], dtype=np.float32)
                # Pad state to 32 dims (rest zero)
                state32 = np.zeros(32, dtype=np.float32)
                state32[:len(state)] = state
                chosen.append({
                    "combined_pil": combined,
                    "front_np": np.asarray(img, dtype=np.uint8),
                    "wrist_np": np.asarray(wrist, dtype=np.uint8),
                    "state": state32,
                    "state_raw": state,
                    "prompt": prompt,
                    "task_index": task_index,
                    "episode_index": int(row["episode_index"][0]),
                    "frame_index": int(row["frame_index"][0]),
                })
    return chosen, prompt


def build_observation(vp, frame, device):
    """Build an Observation object for the action head from a frame dict."""
    obs = {
        "observation/image": frame["front_np"],
        "observation/wrist_image": frame["wrist_np"],
        "observation/state": frame["state_raw"].astype(np.float32),
        "prompt": frame["prompt"],
    }
    import jax
    inputs = jax.tree.map(lambda x: x, obs)
    inputs = vp._input_transform(inputs)
    prompt = inputs["prompt"]
    inputs = vp._policy._remove_strings_transform(inputs)
    inputs = vp.make_wan_images(inputs)
    inputs = jax.tree.map(
        lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...],
        inputs,
    )
    observation = _model.Observation.from_dict(inputs)
    return observation, prompt


def run_v2w_cond(vp, combined_pil, prompt, image_sizes):
    """Conditioned V2W — returns (features, timestep)."""
    vel_pred, vel_ts, _ = vp.video_model.pipe.eval_vid2world(
        prompt=prompt,
        initial_image=combined_pil,
        num_video_frames=NUM_VIDEO_FRAMES,
        resolution=f"{image_sizes[0]},{image_sizes[1]}",
        force_output_frames=True,
        intermediate_feature_ids=[vp.feature_id],
        num_steps=vp.num_global_denoising_steps,
        target_denoising_steps=None,
    )
    feat = vel_pred[vp.inference_denoising_steps][vp.feature_id].clone()
    ts = vel_ts[vp.inference_denoising_steps].unsqueeze(0).to(torch.float32) / 1000.0
    return feat, ts


def run_v2w_uncond(vp, combined_pil, image_sizes, device):
    """Uncond V2W — zero text embedding."""
    pipe = vp.video_model.pipe
    uncond_key = ("__probe_uncond__", None)
    if uncond_key not in pipe._text_embedding_cache:
        zero_emb = torch.zeros(1, 512, 100352, dtype=torch.bfloat16, device=device)
        pipe._text_embedding_cache[uncond_key] = {"emb": zero_emb.cpu(), "neg_emb": zero_emb.cpu()}
    vel_pred, vel_ts, _ = pipe.eval_vid2world(
        prompt="__probe_uncond__",
        initial_image=combined_pil,
        num_video_frames=NUM_VIDEO_FRAMES,
        resolution=f"{image_sizes[0]},{image_sizes[1]}",
        force_output_frames=True,
        intermediate_feature_ids=[vp.feature_id],
        num_steps=vp.num_global_denoising_steps,
        target_denoising_steps=None,
    )
    feat = vel_pred[vp.inference_denoising_steps][vp.feature_id].clone()
    ts = vel_ts[vp.inference_denoising_steps].unsqueeze(0).to(torch.float32) / 1000.0
    return feat, ts


def prepare_features(feat, vp, image_sizes):
    """Reshape to (1, tokens, feature_dim), keep first NUM_LATENT_FRAMES."""
    VIDEO_LATENT_SHAPE = (NUM_LATENT_FRAMES, image_sizes[0] // 16, image_sizes[1] // 16)
    f = feat.view(1, VIDEO_LATENT_SHAPE[0], VIDEO_LATENT_SHAPE[1], VIDEO_LATENT_SHAPE[2], vp.feature_dim)
    return f[:, :NUM_LATENT_FRAMES].reshape(1, -1, vp.feature_dim).clone()


def capture_attention_all_layers(vp, observation, features, ts, noise, device,
                                  denoise_step_idx, num_steps=10):
    """Run action head once, capture attention at *every* Gemma layer."""
    action_model = vp._policy._model
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        result = action_model.visualize_attention(
            device=device,
            observation=observation,
            video_latents=features,
            video_timestep=ts,
            noise=noise,
            denoise_step_idx=denoise_step_idx,
            num_steps=num_steps,
            save_path=None,
            layers=None,  # ALL layers
            head_reduction="mean",
        )
    return result  # seg_attn: (n_layers, n_actions, n_segments), seg_names: [...]


def compute_tr_from_attention(seg_attn, seg_names, action_weighting="mean"):
    """Return TR per layer given raw segment attention.

    seg_attn: (n_layers, n_actions, n_segments)
    Returns: (n_layers,) TR values.
    """
    video_idx = [i for i, n in enumerate(seg_names) if n.startswith("video_t")]
    t0 = video_idx[0]
    future = video_idx[1:]

    if action_weighting == "mean":
        per_seg = seg_attn.mean(axis=1)  # (n_layers, n_segments)
    elif action_weighting == "early":
        per_seg = seg_attn[:, :5].mean(axis=1)
    elif action_weighting == "late":
        per_seg = seg_attn[:, -5:].mean(axis=1)
    else:
        raise ValueError(action_weighting)

    tr = per_seg[:, future].sum(axis=1) / (per_seg[:, t0] + 1e-8)
    return tr.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--ckpt_name", required=True)
    parser.add_argument("--task_ids", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--frames_per_task", type=int, default=6)
    parser.add_argument("--frame_stride", type=int, default=25)
    parser.add_argument("--denoise_steps_captured", type=int, nargs="+", default=[0, 5, 9])
    parser.add_argument("--num_action_denoising_steps", type=int, default=10)
    parser.add_argument("--num_global_denoising_steps", type=int, default=6)
    parser.add_argument("--inference_denoising_steps", type=int, default=2)
    parser.add_argument("--vid2world_mode", default="libero_flat_short")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load policy ──
    print(f"[{time.strftime('%H:%M:%S')}] Loading {args.ckpt_name} on {args.device}")
    pc = _config.get_config("videoVLA_libero_short")
    bp = _policy_config.create_trained_policy(pc, args.ckpt, pytorch_device=args.device)
    vp = _policy.VideoModelPolicyWrapper(
        bp, device=args.device,
        inference_denoising_steps=args.inference_denoising_steps,
        vid2world_mode=args.vid2world_mode,
        feature_id=19,
        num_global_denoising_steps=args.num_global_denoising_steps,
        vid2world_ckpt=args.ckpt,
    )
    image_sizes = IMAGE_SIZES[args.vid2world_mode]
    VIDEO_LATENT_SHAPE = (NUM_LATENT_FRAMES, image_sizes[0] // 16, image_sizes[1] // 16)
    vp._policy._model.set_video_latent_shape(VIDEO_LATENT_SHAPE)

    rng = np.random.RandomState(args.seed)

    # ── For each task, sample frames & probe ──
    meta = {"ckpt": args.ckpt, "ckpt_name": args.ckpt_name,
            "task_ids": args.task_ids, "frames_per_task": args.frames_per_task,
            "denoise_steps_captured": args.denoise_steps_captured,
            "num_action_denoising_steps": args.num_action_denoising_steps}

    all_results = []
    for task_id in args.task_ids:
        print(f"[{time.strftime('%H:%M:%S')}] Task {task_id}: loading frames")
        frames, prompt = load_task_frames(task_id, args.frames_per_task,
                                          frame_stride=args.frame_stride)
        print(f"  Prompt: {prompt[:80]}")
        print(f"  Got {len(frames)} frames")
        for fi, frame in enumerate(frames):
            t0 = time.time()
            observation, frame_prompt = build_observation(vp, frame, args.device)
            # Fresh noise per frame so TR doesn't couple to a shared seed
            noise = torch.from_numpy(
                rng.randn(1, vp._policy._model.action_horizon,
                          vp._policy._model.config.action_dim).astype(np.float32)
            ).to(args.device)

            try:
                # Cond V2W
                feat_cond, ts_cond = run_v2w_cond(vp, frame["combined_pil"], frame_prompt, image_sizes)
                f_cond = prepare_features(feat_cond, vp, image_sizes)
                # Uncond V2W
                feat_uncond, ts_uncond = run_v2w_uncond(vp, frame["combined_pil"], image_sizes, args.device)
                f_uncond = prepare_features(feat_uncond, vp, image_sizes)
            except Exception as e:
                print(f"  V2W failed on task {task_id} frame {fi}: {e}")
                continue

            # Capture attention at selected denoise steps for both cond and uncond
            per_frame = {
                "task_id": task_id, "frame_idx": fi, "prompt": frame_prompt,
                "episode_index": frame["episode_index"],
                "frame_index": frame["frame_index"],
                "cond": {}, "uncond": {},
            }

            feat_cos = float(torch.nn.functional.cosine_similarity(
                f_cond.reshape(1, -1), f_uncond.reshape(1, -1)
            ).item())
            per_frame["feat_cos_cond_uncond"] = feat_cos

            for step_idx in args.denoise_steps_captured:
                try:
                    r_cond = capture_attention_all_layers(
                        vp, observation, f_cond, ts_cond, noise.clone(), args.device,
                        denoise_step_idx=step_idx, num_steps=args.num_action_denoising_steps,
                    )
                    r_uncond = capture_attention_all_layers(
                        vp, observation, f_uncond, ts_uncond, noise.clone(), args.device,
                        denoise_step_idx=step_idx, num_steps=args.num_action_denoising_steps,
                    )
                except Exception as e:
                    print(f"  attn-capture failed step={step_idx}: {e}")
                    continue

                per_frame["cond"][str(step_idx)] = {
                    "seg_attn": r_cond["segment_attention"].astype(np.float32),
                    "seg_names": r_cond["segment_names"],
                }
                per_frame["uncond"][str(step_idx)] = {
                    "seg_attn": r_uncond["segment_attention"].astype(np.float32),
                    "seg_names": r_uncond["segment_names"],
                }

            all_results.append(per_frame)
            print(f"  [{time.strftime('%H:%M:%S')}] task {task_id} frame {fi} "
                  f"({time.time()-t0:.1f}s) feat_cos={feat_cos:.3f}")

    # ── Save ──
    # Save as .npz (float arrays) + meta.json
    save_path = out / "probe.npz"
    packed = {}
    idx_map = []
    for ri, r in enumerate(all_results):
        for sname in ("cond", "uncond"):
            for step in r[sname]:
                key = f"{sname}_step{step}_idx{ri}"
                packed[key] = r[sname][step]["seg_attn"]
                idx_map.append({
                    "array_key": key, "entry_idx": ri,
                    "branch": sname, "denoise_step": int(step),
                })
        packed[f"seg_names_idx{ri}"] = np.array(r["cond"][str(args.denoise_steps_captured[0])]["seg_names"]
                                                 if r["cond"] else [])
    np.savez_compressed(str(save_path), **packed)

    meta["entries"] = [
        {k: v for k, v in r.items() if k not in ("cond", "uncond")}
        for r in all_results
    ]
    meta["idx_map"] = idx_map
    meta["n_entries"] = len(all_results)

    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved to {out}/probe.npz + meta.json — {len(all_results)} frames")


if __name__ == "__main__":
    main()
