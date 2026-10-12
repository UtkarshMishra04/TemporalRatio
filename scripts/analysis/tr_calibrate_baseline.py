#!/usr/bin/env python3
"""Compute a population-level TR baseline per ckpt + recommended robust config.

For each (ckpt), we compute:
  - median TR across 40 probe frames at selected layer (or aggregate of layers)
  - IQR of TR (noise floor we need to beat)
  - recommended tr_baseline value, tr_layer(s), and EMA warm-up period

Writes results/tr_probe/calibration.json:
  {
    ckpt_name: {
      "tr_baseline_L2":   3.54,
      "tr_baseline_L15":  1.23,
      "tr_baseline_avg2_15": 2.38,
      "layer_snr": {2: 1.3, ...},
      "suggested_tr_layer": 2,
      "frame_cv_at_L2": 0.072,
    },
    ...
  }
"""
import json
from pathlib import Path

import numpy as np
from importlib import import_module
_ana = import_module("28_tr_probe_analysis")


def main():
    root = Path("results/tr_probe")
    ckpt_dirs = sorted(d for d in root.iterdir()
                       if d.is_dir() and (d / "probe.npz").exists())
    if not ckpt_dirs:
        print("no probes")
        return

    calib = {}
    for cd in ckpt_dirs:
        data = _ana.analyze_ckpt(cd)
        cond = data["cond_tr"]   # (F, D, L)
        # Average over denoise steps → per-frame TR per layer
        cond_f = np.nanmean(cond, axis=1)   # (F, L)
        uncond_f = np.nanmean(data["uncond_tr"], axis=1)
        ratio_f = cond_f / (uncond_f + 1e-8)

        n_layers = cond.shape[-1]
        layer_info = {}
        for l in range(n_layers):
            baseline = float(np.median(cond_f[:, l]))
            iqr = float(np.subtract(*np.percentile(cond_f[:, l], [75, 25])))
            ratio_mean = float(ratio_f[:, l].mean())
            ratio_std = float(ratio_f[:, l].std())
            snr = (ratio_mean - 1.0) / (ratio_std + 1e-8)
            layer_info[l] = {
                "cond_median": baseline,
                "cond_iqr": iqr,
                "cond_mean": float(cond_f[:, l].mean()),
                "cond_cv": float(cond_f[:, l].std() / (abs(cond_f[:, l].mean()) + 1e-8)),
                "ratio_mean": ratio_mean,
                "ratio_std": ratio_std,
                "ratio_snr": snr,
            }

        # Aggregates: mean of {L2, L15}  (a reasonable compromise between v2 and v3)
        agg_keys = [2, 15]
        cond_agg = cond_f[:, agg_keys].mean(axis=1)
        uncond_agg = uncond_f[:, agg_keys].mean(axis=1)
        ratio_agg = cond_agg / (uncond_agg + 1e-8)
        agg_info = {
            "layers": agg_keys,
            "cond_median": float(np.median(cond_agg)),
            "cond_iqr": float(np.subtract(*np.percentile(cond_agg, [75, 25]))),
            "ratio_mean": float(ratio_agg.mean()),
            "ratio_std": float(ratio_agg.std()),
            "ratio_snr": float((ratio_agg.mean() - 1.0) / (ratio_agg.std() + 1e-8)),
        }

        # Pick best single-layer by SNR  (no hard-coded default)
        best_l = int(max(layer_info.keys(), key=lambda l: layer_info[l]["ratio_snr"]))

        calib[cd.name] = {
            "layers": layer_info,
            "agg_L2_L15": agg_info,
            "suggested_tr_layer": best_l,
            "suggested_tr_baseline": layer_info[best_l]["cond_median"],
            "best_snr": layer_info[best_l]["ratio_snr"],
        }

    out = Path("results/tr_probe/calibration.json")
    with open(out, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"Wrote {out}")

    # Pretty print per-ckpt recommendation
    print("\n" + "=" * 100)
    print("PER-CHECKPOINT CALIBRATION RECOMMENDATION")
    print("=" * 100)
    print(f"{'ckpt':<20}  {'best_L':>7}  {'SNR':>5}  {'baseline':>9}  "
          f"{'agg(2,15)_SNR':>13}  {'agg_baseline':>12}")
    for name, c in calib.items():
        bl = c["suggested_tr_layer"]
        li = c["layers"][bl]
        agg = c["agg_L2_L15"]
        print(f"{name:<20}  L{bl:<5}  {li['ratio_snr']:>5.2f}  {li['cond_median']:>9.3f}  "
              f"{agg['ratio_snr']:>13.2f}  {agg['cond_median']:>12.3f}")


if __name__ == "__main__":
    main()
