#!/usr/bin/env python3
"""Analyze TR probe outputs from 27_tr_probe.py.

For each checkpoint, load probe.npz + meta.json and compute:
  - Per-layer TR (mean over actions, over frames) for cond / uncond.
  - Cond-vs-uncond TR ratio per layer (a language-sensitivity proxy).
  - Across-frame CV (noise) of the per-layer TR.
  - Per-denoise-step stability: does TR move with denoise step?
  - Per-action-position TR pattern (early actions vs late actions).

Outputs:
  - results/tr_probe/summary.json: aggregate stats per (ckpt, layer, branch).
  - Prints a leaderboard:
      layer,
      mean cond TR, mean uncond TR, cond/uncond ratio,
      across-frame CV of cond TR,
      per-denoise CV of cond TR.
  - Identifies "best" layers by (high ratio, low CV).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def tr_from_seg_attn(seg_attn: np.ndarray, seg_names, action_weighting="mean"):
    """seg_attn: (L, A, S). Returns per-layer TR (L,)."""
    video_idx = [i for i, n in enumerate(seg_names) if n.startswith("video_t")]
    t0 = video_idx[0]
    future = video_idx[1:]

    if action_weighting == "mean":
        per_seg = seg_attn.mean(axis=1)
    elif action_weighting == "early":
        per_seg = seg_attn[:, :5].mean(axis=1)
    elif action_weighting == "late":
        per_seg = seg_attn[:, -5:].mean(axis=1)
    else:
        raise ValueError(action_weighting)

    tr = per_seg[:, future].sum(axis=1) / (per_seg[:, t0] + 1e-8)
    return tr.astype(np.float32)


def load_ckpt_probe(ckpt_dir: Path):
    """Load probe.npz + meta.json for one ckpt."""
    npz = np.load(str(ckpt_dir / "probe.npz"))
    meta = json.loads((ckpt_dir / "meta.json").read_text())
    return npz, meta


def analyze_ckpt(ckpt_dir: Path, action_weighting="mean"):
    """Compute per-layer stats for a single checkpoint.

    Returns dict: {
      layers: n_layers,
      denoise_steps: [...],
      cond_tr:    (n_frames, n_denoise, n_layers),
      uncond_tr:  (n_frames, n_denoise, n_layers),
      seg_names: [...]
    }
    """
    npz, meta = load_ckpt_probe(ckpt_dir)
    n_entries = meta["n_entries"]
    denoise_steps = sorted(set(int(e["denoise_step"]) for e in meta["idx_map"]))

    # Probe arrays share (layers, actions, segments). Pull shape.
    first_key = meta["idx_map"][0]["array_key"]
    n_layers, n_actions, n_segments = npz[first_key].shape
    seg_names_any = None
    for ri in range(n_entries):
        k = f"seg_names_idx{ri}"
        if k in npz.files and len(npz[k]):
            seg_names_any = list(npz[k])
            break

    cond = np.full((n_entries, len(denoise_steps), n_layers), np.nan, dtype=np.float32)
    uncond = np.full_like(cond, np.nan)

    # Also per-action TR (a full break down): (n_entries, n_denoise, n_layers, n_actions)
    cond_per_action = np.full((n_entries, len(denoise_steps), n_layers, n_actions),
                              np.nan, dtype=np.float32)
    uncond_per_action = np.full_like(cond_per_action, np.nan)

    ds_index = {s: i for i, s in enumerate(denoise_steps)}
    seg_names = seg_names_any
    t0 = None
    future = None
    if seg_names is not None:
        video_idx = [i for i, n in enumerate(seg_names) if n.startswith("video_t")]
        t0 = video_idx[0]
        future = video_idx[1:]

    for item in meta["idx_map"]:
        k = item["array_key"]
        ri = item["entry_idx"]
        st = int(item["denoise_step"])
        arr = npz[k]  # (L, A, S)

        if t0 is None:
            video_idx = [i for i, n in enumerate(seg_names) if n.startswith("video_t")]
            t0 = video_idx[0]
            future = video_idx[1:]

        tr = tr_from_seg_attn(arr, seg_names, action_weighting)

        # per-action version
        per_action_tr = arr[..., future].sum(axis=-1) / (arr[..., t0] + 1e-8)  # (L, A)

        if item["branch"] == "cond":
            cond[ri, ds_index[st]] = tr
            cond_per_action[ri, ds_index[st]] = per_action_tr
        else:
            uncond[ri, ds_index[st]] = tr
            uncond_per_action[ri, ds_index[st]] = per_action_tr

    return {
        "n_layers": n_layers,
        "n_actions": n_actions,
        "n_entries": n_entries,
        "denoise_steps": denoise_steps,
        "cond_tr": cond,  # (n_entries, n_denoise, n_layers)
        "uncond_tr": uncond,
        "cond_per_action": cond_per_action,
        "uncond_per_action": uncond_per_action,
        "seg_names": seg_names,
        "meta": meta,
    }


def per_layer_stats(data):
    """Aggregate cond/uncond TR into per-layer metrics."""
    cond = data["cond_tr"]  # (F, D, L)
    uncond = data["uncond_tr"]

    # Average over denoise steps per frame
    cond_frames = np.nanmean(cond, axis=1)   # (F, L)
    uncond_frames = np.nanmean(uncond, axis=1)

    cond_mean = np.nanmean(cond_frames, axis=0)  # (L,)
    cond_std = np.nanstd(cond_frames, axis=0)
    cond_cv = cond_std / (np.abs(cond_mean) + 1e-8)

    uncond_mean = np.nanmean(uncond_frames, axis=0)
    uncond_std = np.nanstd(uncond_frames, axis=0)

    # Language sensitivity: ratio cond/uncond (>1 means language raises future attention)
    ratio = cond_mean / (uncond_mean + 1e-8)

    # Per-denoise-step CV per layer (averaged over frames first)
    cond_per_ds = np.nanmean(cond, axis=0)  # (D, L)
    denoise_cv = np.nanstd(cond_per_ds, axis=0) / (
        np.abs(np.nanmean(cond_per_ds, axis=0)) + 1e-8
    )

    return {
        "cond_mean": cond_mean,
        "cond_std": cond_std,
        "cond_cv": cond_cv,
        "uncond_mean": uncond_mean,
        "uncond_std": uncond_std,
        "ratio": ratio,
        "denoise_cv": denoise_cv,
    }


def score_layers(stats):
    """Layer quality score = (cond TR magnitude) × (cond vs uncond separation) / (noise).

    Interprets:
      - larger cond TR magnitude => more responsive to future video.
      - larger cond/uncond ratio => language-driven signal.
      - smaller across-frame CV => reproducible signal.
    """
    mag = np.log1p(np.clip(stats["cond_mean"], 0, None))
    sep = np.log(np.clip(stats["ratio"], 1e-3, None))
    noise = stats["cond_cv"] + stats["denoise_cv"]
    # Use harmonic-ish combination so noisy layers score low.
    return mag + 0.5 * sep - np.clip(noise, 0, 5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/tr_probe")
    parser.add_argument("--action_weighting", default="mean",
                        choices=["mean", "early", "late"])
    parser.add_argument("--out", default="results/tr_probe/summary.json")
    args = parser.parse_args()

    root = Path(args.root)
    ckpt_dirs = sorted(d for d in root.iterdir()
                       if d.is_dir() and (d / "probe.npz").exists())
    if not ckpt_dirs:
        print(f"No probe.npz under {root}")
        return

    print(f"Found {len(ckpt_dirs)} probes")
    summary = {}
    per_ckpt_stats = {}

    for ckpt_dir in ckpt_dirs:
        name = ckpt_dir.name
        print(f"\n[{name}] loading...")
        data = analyze_ckpt(ckpt_dir, args.action_weighting)
        stats = per_layer_stats(data)
        summary[name] = {
            "n_entries": int(data["n_entries"]),
            "denoise_steps": data["denoise_steps"],
            "layers": {
                l: {
                    "cond_mean": float(stats["cond_mean"][l]),
                    "cond_std": float(stats["cond_std"][l]),
                    "cond_cv": float(stats["cond_cv"][l]),
                    "uncond_mean": float(stats["uncond_mean"][l]),
                    "ratio": float(stats["ratio"][l]),
                    "denoise_cv": float(stats["denoise_cv"][l]),
                } for l in range(data["n_layers"])
            },
        }
        per_ckpt_stats[name] = {"data": data, "stats": stats}

    # Save JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {out_path}")

    # Pretty leaderboard
    print("\n" + "=" * 100)
    print("PER-CKPT PER-LAYER TABLE — TR with action_weighting=mean")
    print("=" * 100)
    print(f"{'ckpt':<20} {'L':>3} {'cond_μ':>8} {'uncnd_μ':>8} {'ratio':>7} {'cond_CV':>8} {'ds_CV':>6} {'score':>7}")
    for name, sts in per_ckpt_stats.items():
        stats = sts["stats"]
        scores = score_layers(stats)
        n_layers = len(stats["cond_mean"])
        # Print top 6 layers per ckpt by score
        top = np.argsort(-scores)[:6]
        for l in sorted(top):
            print(f"{name:<20} {l:>3} "
                  f"{stats['cond_mean'][l]:>8.3f} "
                  f"{stats['uncond_mean'][l]:>8.3f} "
                  f"{stats['ratio'][l]:>7.3f} "
                  f"{stats['cond_cv'][l]:>8.3f} "
                  f"{stats['denoise_cv'][l]:>6.3f} "
                  f"{scores[l]:>7.3f}")
        print("-" * 100)

    # Cross-ckpt best layer
    print("\n=== BEST LAYER CONSENSUS ACROSS CKPTS ===")
    # Stack scores across ckpts (layers aligned by index)
    names = list(per_ckpt_stats.keys())
    score_mat = np.stack([score_layers(per_ckpt_stats[n]["stats"]) for n in names])
    mean_score = score_mat.mean(axis=0)
    # Separately: mean ratio (language sensitivity) and mean CV
    ratios = np.stack([per_ckpt_stats[n]["stats"]["ratio"] for n in names])
    cvs = np.stack([per_ckpt_stats[n]["stats"]["cond_cv"] for n in names])
    print(f"{'L':>3} {'score':>7} {'mean_ratio':>10} {'mean_CV':>8}")
    order = np.argsort(-mean_score)
    for l in order[:8]:
        print(f"{l:>3} {mean_score[l]:>7.3f} {ratios[:, l].mean():>10.3f} {cvs[:, l].mean():>8.3f}")

    # Additional: per-action breakdown at best layer
    best_l = int(order[0])
    print(f"\n=== PER-ACTION TR AT LAYER {best_l} (averaged over frames, over denoise steps) ===")
    print(f"{'ckpt':<20} " + " ".join(f"a{i:>2}" for i in range(17)))
    for name in names:
        d = per_ckpt_stats[name]["data"]
        per_action_cond = np.nanmean(d["cond_per_action"], axis=(0, 1))  # (L, A)
        row = per_action_cond[best_l]
        print(f"{name:<20} " + " ".join(f"{v:>4.2f}" for v in row))


if __name__ == "__main__":
    main()
