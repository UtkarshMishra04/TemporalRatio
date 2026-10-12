"""Per-suite TR analysis across every model × cfg config.

Outputs:
1. CSV table: suite × model × cfg → TR_0, TR_mean, TR_end, TR_slope, SR%, n_episodes
2. Per-suite TR curve plots (one subplot per suite, curves per config)
3. Suite-level heatmaps showing TR dynamics differences
"""
import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import csv

import argparse

_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_parser.add_argument("--results", required=True, help="LIBERO evaluation output root containing **/tr_logs")
_parser.add_argument("--out", default="results/reports/tr_tables", help="where to write the CSV and plots")
_args = _parser.parse_args()
BASE = _args.results
OUT = _args.out
os.makedirs(OUT, exist_ok=True)

# Auto-discover all model/step/config combos from tr_logs directories
def discover_configs():
    """Walk tr_logs dirs and extract (model, step, config, suite, task, path)."""
    pattern = os.path.join(BASE, "**/tr_logs")
    tr_dirs = glob.glob(pattern, recursive=True)

    configs = defaultdict(lambda: defaultdict(list))  # (model_cfg) -> suite -> [episodes]

    for tr_dir in sorted(tr_dirs):
        # Parse path: BASE/model/step/config/.../suite/task_N/tr_logs
        rel = os.path.relpath(tr_dir, BASE)
        parts = rel.split('/')

        # Extract model/step/config from first 3 parts
        model_step = parts[0] + '/' + parts[1]
        config = parts[2]

        # Extract suite from path (look for libero_* directory)
        suite = None
        for p in parts:
            if p.startswith('libero_'):
                suite = p
        if suite is None:
            continue

        # Load all episode TR logs
        ep_files = sorted(glob.glob(os.path.join(tr_dir, "ep_*_tr.json")))
        for ef in ep_files:
            try:
                d = json.load(open(ef))
                # Override suite from JSON if available, else use dir
                d.setdefault('suite', suite)
                configs[(model_step, config)][suite].append(d)
            except:
                continue

    return configs


def get_tr_stats(episodes, max_replans=25):
    """Compute TR statistics for a set of episodes."""
    tr_curves = []
    tr0s = []
    successes = 0

    for ep in episodes:
        if ep.get('success', False):
            successes += 1
        traj = ep.get('trajectory', [])
        if len(traj) < 3:
            continue
        trs = [s.get('tr_k', ep.get('tr_0', 1)) for s in traj[:max_replans]]
        tr0s.append(ep.get('tr_0', trs[0] if trs else 1))
        while len(trs) < max_replans:
            trs.append(trs[-1])
        tr_curves.append(trs)

    if not tr_curves:
        return None

    arr = np.array(tr_curves)
    mean_curve = arr.mean(axis=0)
    n = len(episodes)
    sr = successes / n * 100 if n > 0 else 0

    # Compute slope (linear fit on first 15 replans)
    x = np.arange(min(15, len(mean_curve)))
    if len(x) > 2:
        slope = np.polyfit(x, mean_curve[:len(x)], 1)[0]
    else:
        slope = 0

    return {
        'n': n,
        'n_valid': len(tr_curves),
        'sr': sr,
        'tr_0': mean_curve[0],
        'tr_mean': mean_curve[:15].mean(),
        'tr_end': mean_curve[-5:].mean(),
        'tr_peak': mean_curve[:15].max(),
        'tr_min': mean_curve[:15].min(),
        'tr_slope': slope,
        'tr_std_0': np.array(tr0s).std() if tr0s else 0,
        'mean_curve': mean_curve,
    }


# Friendly names
MODEL_NAMES = {
    'libero_2b_lora_joint/76000': 'Joint_76k',
    'joint_93f_lora_s3_action/40000': '93f_S3',
    'joint_93f_lora_s2_joint/30000': '93f_S2',
    'libero_lora_2stage_L19_s2_action/60000': '2S_r32',
    'libero_lora_r64_2stage_L19_s2_action/60000': '2S_r64_do',
    'libero_lora_r128_2stage_L19_s2_action/60000': '2S_r128_do',
}

SUITE_ORDER = [
    'libero_spatial', 'libero_object', 'libero_goal', 'libero_10',
    'libero_spatial_ood', 'libero_object_ood', 'libero_goal_ood', 'libero_90',
]

SUITE_SHORT = {
    'libero_spatial': 'Spatial',
    'libero_object': 'Object',
    'libero_goal': 'Goal',
    'libero_10': 'L10',
    'libero_spatial_ood': 'Spatial_OOD',
    'libero_object_ood': 'Object_OOD',
    'libero_goal_ood': 'Goal_OOD',
    'libero_90': 'L90',
}


print("Discovering configs...")
configs = discover_configs()
print(f"Found {len(configs)} model×config combos")

# Compute stats for every (model_cfg, suite) pair
all_stats = {}  # (model_step, config, suite) -> stats
for (model_step, config), suite_eps in configs.items():
    for suite, episodes in suite_eps.items():
        stats = get_tr_stats(episodes)
        if stats:
            all_stats[(model_step, config, suite)] = stats

print(f"Computed stats for {len(all_stats)} (model, config, suite) triples")

# ========================================
# OUTPUT 1: CSV table
# ========================================
csv_path = f'{OUT}/per_suite_tr_table.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Model', 'Config', 'Suite', 'N', 'SR%', 'TR_0', 'TR_mean', 'TR_end', 'TR_peak', 'TR_min', 'TR_slope', 'TR_std0'])
    for (ms, cfg, suite), st in sorted(all_stats.items()):
        mname = MODEL_NAMES.get(ms, ms)
        w.writerow([mname, cfg, suite, st['n'], f"{st['sr']:.1f}",
                     f"{st['tr_0']:.2f}", f"{st['tr_mean']:.2f}", f"{st['tr_end']:.2f}",
                     f"{st['tr_peak']:.2f}", f"{st['tr_min']:.2f}",
                     f"{st['tr_slope']:.4f}", f"{st['tr_std_0']:.2f}"])
print(f"\nCSV saved: {csv_path}")

# ========================================
# OUTPUT 2: Printed summary table
# ========================================
print("\n" + "=" * 130)
print("PER-SUITE TR DYNAMICS")
print("=" * 130)
print(f"{'Model':<14} {'Config':<25} {'Suite':<16} {'N':>3} {'SR%':>5} {'TR_0':>6} {'TR_m':>6} {'TR_end':>6} {'Slope':>7} {'TR_std':>6}")
print("-" * 130)

# Group by model for cleaner output
by_model = defaultdict(list)
for (ms, cfg, suite), st in sorted(all_stats.items()):
    by_model[ms].append((cfg, suite, st))

for ms in sorted(by_model.keys()):
    mname = MODEL_NAMES.get(ms, ms)
    entries = by_model[ms]
    # Sort by config then suite
    entries.sort(key=lambda x: (x[0], SUITE_ORDER.index(x[1]) if x[1] in SUITE_ORDER else 99))
    prev_cfg = None
    for cfg, suite, st in entries:
        if cfg != prev_cfg and prev_cfg is not None:
            print()  # blank line between configs
        prev_cfg = cfg
        sname = SUITE_SHORT.get(suite, suite)
        print(f"{mname:<14} {cfg:<25} {sname:<16} {st['n']:>3} {st['sr']:>5.1f} {st['tr_0']:>6.2f} {st['tr_mean']:>6.2f} {st['tr_end']:>6.2f} {st['tr_slope']:>7.4f} {st['tr_std_0']:>6.2f}")
    print()

# ========================================
# OUTPUT 3: Per-suite TR curves for each model (one figure per model)
# ========================================
print("\nGenerating per-suite TR curve plots...")

for ms in sorted(by_model.keys()):
    mname = MODEL_NAMES.get(ms, ms)
    entries = by_model[ms]

    # Get unique suites and configs
    suites_present = sorted(set(s for _, s, _ in entries),
                           key=lambda x: SUITE_ORDER.index(x) if x in SUITE_ORDER else 99)
    configs_present = sorted(set(c for c, _, _ in entries))

    if len(suites_present) < 2:
        continue

    n_suites = len(suites_present)
    ncols = min(4, n_suites)
    nrows = (n_suites + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    # Color map for configs
    cmap = plt.cm.tab10
    cfg_colors = {c: cmap(i / max(1, len(configs_present) - 1)) for i, c in enumerate(configs_present)}

    for idx, suite in enumerate(suites_present):
        ax = axes[idx // ncols][idx % ncols]
        sname = SUITE_SHORT.get(suite, suite)

        for cfg in configs_present:
            key = (ms, cfg, suite)
            if key not in all_stats:
                continue
            st = all_stats[key]
            curve = st['mean_curve']
            label = f"{cfg} (SR={st['sr']:.0f}%, n={st['n']})"
            ax.plot(range(len(curve)), curve, color=cfg_colors[cfg],
                    label=label, linewidth=1.5, alpha=0.8)

        ax.set_title(sname, fontsize=10)
        ax.set_xlabel('Replan #', fontsize=8)
        ax.set_ylabel('TR', fontsize=8)
        ax.legend(fontsize=6, loc='best')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_suites, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f'{mname}: TR Curves by Suite and Config', fontsize=13)
    fig.tight_layout()
    safe_name = mname.replace('/', '_').replace(' ', '_')
    fig.savefig(f'{OUT}/18_tr_per_suite_{safe_name}.png', dpi=150)
    plt.close()
    print(f"  Saved: 18_tr_per_suite_{safe_name}.png")

# ========================================
# OUTPUT 4: Suite comparison across models (for matching configs)
# ========================================
print("\nGenerating cross-model suite comparison...")

# Find configs that appear in multiple models
cfg_models = defaultdict(set)
for (ms, cfg, suite), _ in all_stats.items():
    cfg_models[cfg].add(ms)

# Use baseline as the universal comparison
common_cfgs = ['baseline']
# Also check for common non-baseline configs
for cfg in sorted(cfg_models.keys()):
    if len(cfg_models[cfg]) >= 2 and cfg not in common_cfgs:
        # Normalize config names that mean the same thing
        if 'lang01' in cfg:
            common_cfgs.append(cfg)

for compare_cfg_pattern in ['baseline', 'lang01']:
    # Find all configs matching this pattern
    matching = {}
    for (ms, cfg, suite), st in all_stats.items():
        if compare_cfg_pattern in cfg or (compare_cfg_pattern == 'baseline' and 'baseline' in cfg):
            matching.setdefault(ms, {})[suite] = (cfg, st)

    if len(matching) < 2:
        continue

    fig, axes = plt.subplots(2, 4, figsize=(20, 8), squeeze=False)

    model_colors = {}
    cmap = plt.cm.Set1
    for i, ms in enumerate(sorted(matching.keys())):
        model_colors[ms] = cmap(i / max(1, len(matching) - 1))

    for idx, suite in enumerate(SUITE_ORDER):
        if idx >= 8:
            break
        ax = axes[idx // 4][idx % 4]
        sname = SUITE_SHORT.get(suite, suite)

        for ms in sorted(matching.keys()):
            if suite not in matching[ms]:
                continue
            cfg, st = matching[ms][suite]
            mname = MODEL_NAMES.get(ms, ms)
            curve = st['mean_curve']
            ax.plot(range(len(curve)), curve, color=model_colors[ms],
                    label=f"{mname} (SR={st['sr']:.0f}%)", linewidth=2, alpha=0.8)

        ax.set_title(sname, fontsize=10)
        ax.set_xlabel('Replan #', fontsize=8)
        ax.set_ylabel('TR', fontsize=8)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Cross-Model TR by Suite ({compare_cfg_pattern} configs)', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{OUT}/19_cross_model_suite_{compare_cfg_pattern}.png', dpi=150)
    plt.close()
    print(f"  Saved: 19_cross_model_suite_{compare_cfg_pattern}.png")

# ========================================
# OUTPUT 5: TR heatmap — suite × config for each model
# ========================================
print("\nGenerating TR heatmaps...")

for ms in sorted(by_model.keys()):
    mname = MODEL_NAMES.get(ms, ms)
    entries = by_model[ms]

    suites_present = sorted(set(s for _, s, _ in entries),
                           key=lambda x: SUITE_ORDER.index(x) if x in SUITE_ORDER else 99)
    configs_present = sorted(set(c for c, _, _ in entries))

    if len(suites_present) < 2 or len(configs_present) < 2:
        continue

    # Build matrices for TR_0, SR, and TR_slope
    for metric_name, metric_key in [('TR_0', 'tr_0'), ('SR', 'sr'), ('TR_slope', 'tr_slope')]:
        matrix = np.full((len(configs_present), len(suites_present)), np.nan)
        for i, cfg in enumerate(configs_present):
            for j, suite in enumerate(suites_present):
                key = (ms, cfg, suite)
                if key in all_stats:
                    matrix[i, j] = all_stats[key][metric_key]

        fig, ax = plt.subplots(figsize=(max(8, len(suites_present) * 1.5), max(4, len(configs_present) * 0.6)))

        if metric_key == 'sr':
            vmin, vmax = 0, 100
            cmap_name = 'RdYlGn'
        elif metric_key == 'tr_slope':
            absmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 0.01)
            vmin, vmax = -absmax, absmax
            cmap_name = 'RdBu_r'
        else:
            vmin, vmax = np.nanmin(matrix), np.nanmax(matrix)
            cmap_name = 'viridis'

        im = ax.imshow(matrix, aspect='auto', cmap=cmap_name, vmin=vmin, vmax=vmax)

        suite_labels = [SUITE_SHORT.get(s, s) for s in suites_present]
        ax.set_xticks(range(len(suites_present)))
        ax.set_xticklabels(suite_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(configs_present)))
        ax.set_yticklabels(configs_present, fontsize=8)

        # Annotate cells
        for i in range(len(configs_present)):
            for j in range(len(suites_present)):
                if not np.isnan(matrix[i, j]):
                    fmt = '.0f' if metric_key == 'sr' else '.2f' if metric_key == 'tr_0' else '.3f'
                    txt = format(matrix[i, j], fmt)
                    ax.text(j, i, txt, ha='center', va='center', fontsize=7,
                           color='white' if abs(matrix[i, j] - vmin) > 0.6 * (vmax - vmin) else 'black')

        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f'{mname}: {metric_name} by Suite × Config', fontsize=11)
        fig.tight_layout()
        safe_name = mname.replace('/', '_').replace(' ', '_')
        fig.savefig(f'{OUT}/20_heatmap_{metric_name}_{safe_name}.png', dpi=150)
        plt.close()

    print(f"  Saved heatmaps for {mname}")

# ========================================
# OUTPUT 6: ID vs OOD TR comparison per model×config
# ========================================
print("\nGenerating ID vs OOD TR comparison...")

# Pair ID and OOD suites
ID_OOD_PAIRS = [
    ('libero_spatial', 'libero_spatial_ood'),
    ('libero_object', 'libero_object_ood'),
    ('libero_goal', 'libero_goal_ood'),
]

rows = []
print("\n" + "=" * 120)
print("ID vs OOD TR COMPARISON")
print("=" * 120)
print(f"{'Model':<14} {'Config':<25} {'Suite':<10} {'ID_TR0':>7} {'OOD_TR0':>8} {'Delta':>6} {'ID_SR':>5} {'OOD_SR':>6} {'ID_slope':>8} {'OOD_slope':>9}")
print("-" * 120)

for ms in sorted(by_model.keys()):
    mname = MODEL_NAMES.get(ms, ms)
    entries = by_model[ms]
    configs_present = sorted(set(c for c, _, _ in entries))

    for cfg in configs_present:
        for id_suite, ood_suite in ID_OOD_PAIRS:
            id_key = (ms, cfg, id_suite)
            ood_key = (ms, cfg, ood_suite)
            if id_key in all_stats and ood_key in all_stats:
                id_st = all_stats[id_key]
                ood_st = all_stats[ood_key]
                delta = ood_st['tr_0'] - id_st['tr_0']
                suite_short = id_suite.replace('libero_', '')
                print(f"{mname:<14} {cfg:<25} {suite_short:<10} {id_st['tr_0']:>7.2f} {ood_st['tr_0']:>8.2f} {delta:>+6.2f} {id_st['sr']:>5.0f} {ood_st['sr']:>6.0f} {id_st['tr_slope']:>8.4f} {ood_st['tr_slope']:>9.4f}")

# ========================================
# OUTPUT 7: Success vs failure TR curves per suite
# ========================================
print("\n\nGenerating success vs failure TR curves per suite...")

for ms in sorted(by_model.keys()):
    mname = MODEL_NAMES.get(ms, ms)
    entries = by_model[ms]

    # Use baseline config for success/failure analysis
    baseline_cfg = None
    for c, _, _ in entries:
        if 'baseline' in c:
            baseline_cfg = c
            break
    if baseline_cfg is None:
        continue

    suites_for_model = sorted(set(s for c, s, _ in entries if c == baseline_cfg),
                              key=lambda x: SUITE_ORDER.index(x) if x in SUITE_ORDER else 99)

    if len(suites_for_model) < 2:
        continue

    n_suites = len(suites_for_model)
    ncols = min(4, n_suites)
    nrows = (n_suites + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, suite in enumerate(suites_for_model):
        ax = axes[idx // ncols][idx % ncols]
        sname = SUITE_SHORT.get(suite, suite)

        # Get episodes for this suite
        key = (ms, baseline_cfg)
        eps = configs.get(key, {}).get(suite, [])

        succ = [e for e in eps if e.get('success', False)]
        fail = [e for e in eps if not e.get('success', False)]

        for group, label, color in [(succ, 'Success', 'green'), (fail, 'Failure', 'red')]:
            st = get_tr_stats(group)
            if st and st['n_valid'] >= 2:
                curve = st['mean_curve']
                ax.plot(range(len(curve)), curve, color=color,
                        label=f"{label} (n={st['n']})", linewidth=2, alpha=0.8)
                # Add std band
                # Recompute for std
                tr_curves = []
                for ep in group:
                    traj = ep.get('trajectory', [])
                    if len(traj) < 3:
                        continue
                    trs = [s.get('tr_k', ep.get('tr_0', 1)) for s in traj[:25]]
                    while len(trs) < 25:
                        trs.append(trs[-1])
                    tr_curves.append(trs)
                if tr_curves:
                    arr = np.array(tr_curves)
                    std = arr.std(axis=0)
                    ax.fill_between(range(len(curve)), curve - std, curve + std,
                                   color=color, alpha=0.15)

        ax.set_title(f'{sname}', fontsize=10)
        ax.set_xlabel('Replan #', fontsize=8)
        ax.set_ylabel('TR', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n_suites, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f'{mname} Baseline: Success vs Failure TR by Suite', fontsize=13)
    fig.tight_layout()
    safe_name = mname.replace('/', '_').replace(' ', '_')
    fig.savefig(f'{OUT}/21_succ_fail_tr_{safe_name}.png', dpi=150)
    plt.close()
    print(f"  Saved: 21_succ_fail_tr_{safe_name}.png")

# ========================================
# OUTPUT 8: CFG effect delta (config - baseline) per suite
# ========================================
print("\nGenerating CFG effect delta analysis...")

print("\n" + "=" * 130)
print("CFG EFFECT ON TR (Delta from Baseline)")
print("=" * 130)
print(f"{'Model':<14} {'Config':<25} {'Suite':<16} {'dTR_0':>7} {'dTR_mean':>8} {'dTR_end':>8} {'dSR':>6} {'dSlope':>7}")
print("-" * 130)

for ms in sorted(by_model.keys()):
    mname = MODEL_NAMES.get(ms, ms)
    entries = by_model[ms]

    # Find baseline
    baseline_cfg = None
    for c, _, _ in entries:
        if 'baseline' in c:
            baseline_cfg = c
            break
    if baseline_cfg is None:
        continue

    configs_present = sorted(set(c for c, _, _ in entries))
    suites_present = sorted(set(s for _, s, _ in entries),
                           key=lambda x: SUITE_ORDER.index(x) if x in SUITE_ORDER else 99)

    for cfg in configs_present:
        if cfg == baseline_cfg:
            continue
        for suite in suites_present:
            base_key = (ms, baseline_cfg, suite)
            cfg_key = (ms, cfg, suite)
            if base_key in all_stats and cfg_key in all_stats:
                b = all_stats[base_key]
                c = all_stats[cfg_key]
                sname = SUITE_SHORT.get(suite, suite)
                dtr0 = c['tr_0'] - b['tr_0']
                dtrm = c['tr_mean'] - b['tr_mean']
                dtre = c['tr_end'] - b['tr_end']
                dsr = c['sr'] - b['sr']
                dslope = c['tr_slope'] - b['tr_slope']
                print(f"{mname:<14} {cfg:<25} {sname:<16} {dtr0:>+7.2f} {dtrm:>+8.2f} {dtre:>+8.2f} {dsr:>+6.1f} {dslope:>+7.4f}")
        print()

# ========================================
# OUTPUT 9: 93f S3 vs S2 direct comparison
# ========================================
print("\n\nGenerating 93f S3 vs S2 comparison...")

S3_KEY = 'joint_93f_lora_s3_action/40000'
S2_KEY = 'joint_93f_lora_s2_joint/30000'

# Find common suites between S3 and S2 baselines
s3_suites = {s for (ms, cfg, s) in all_stats if ms == S3_KEY}
s2_suites = {s for (ms, cfg, s) in all_stats if ms == S2_KEY}
common_93f_suites = sorted(s3_suites & s2_suites,
                           key=lambda x: SUITE_ORDER.index(x) if x in SUITE_ORDER else 99)

if common_93f_suites:
    n_suites = len(common_93f_suites)
    ncols = min(4, n_suites)
    nrows = (n_suites + ncols - 1) // ncols

    # Plot 1: Baseline TR curves S3 vs S2
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for idx, suite in enumerate(common_93f_suites):
        ax = axes[idx // ncols][idx % ncols]
        sname = SUITE_SHORT.get(suite, suite)

        for ms, color, label in [(S2_KEY, 'blue', '93f_S2'), (S3_KEY, 'red', '93f_S3')]:
            # Find baseline config for this model
            baseline_key = None
            for (m, c, s), st in all_stats.items():
                if m == ms and s == suite and 'baseline' in c:
                    baseline_key = (m, c, s)
                    break
            if baseline_key and baseline_key in all_stats:
                st = all_stats[baseline_key]
                curve = st['mean_curve']
                ax.plot(range(len(curve)), curve, color=color,
                        label=f"{label} (SR={st['sr']:.0f}%, n={st['n']})", linewidth=2)

        ax.set_title(sname, fontsize=10)
        ax.set_xlabel('Replan #', fontsize=8)
        ax.set_ylabel('TR', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n_suites, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle('93f S3 vs S2: Baseline TR Curves by Suite', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{OUT}/22_93f_s3_vs_s2_baseline.png', dpi=150)
    plt.close()
    print("  Saved: 22_93f_s3_vs_s2_baseline.png")

    # Plot 2: S3 vs S2 success/failure breakdown
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for idx, suite in enumerate(common_93f_suites):
        ax = axes[idx // ncols][idx % ncols]
        sname = SUITE_SHORT.get(suite, suite)

        for ms, base_color, label in [(S2_KEY, 'blue', 'S2'), (S3_KEY, 'red', 'S3')]:
            baseline_cfg = None
            for (m, c, s) in all_stats:
                if m == ms and s == suite and 'baseline' in c:
                    baseline_cfg = c
                    break
            if baseline_cfg is None:
                continue

            eps = configs.get((ms, baseline_cfg), {}).get(suite, [])
            succ = [e for e in eps if e.get('success', False)]
            fail = [e for e in eps if not e.get('success', False)]

            for group, suffix, ls in [(succ, 'succ', '-'), (fail, 'fail', '--')]:
                st = get_tr_stats(group)
                if st and st['n_valid'] >= 2:
                    curve = st['mean_curve']
                    ax.plot(range(len(curve)), curve, color=base_color, linestyle=ls,
                            label=f"{label}_{suffix} (n={st['n']})", linewidth=1.5, alpha=0.8)

        ax.set_title(sname, fontsize=10)
        ax.set_xlabel('Replan #', fontsize=8)
        ax.set_ylabel('TR', fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    for idx in range(n_suites, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle('93f S3 vs S2: Success/Failure TR by Suite', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{OUT}/23_93f_s3_vs_s2_succ_fail.png', dpi=150)
    plt.close()
    print("  Saved: 23_93f_s3_vs_s2_succ_fail.png")

# Print 93f S3 vs S2 numeric comparison
print("\n" + "=" * 120)
print("93f S3 vs S2 COMPARISON (Baseline)")
print("=" * 120)
print(f"{'Suite':<16} {'S2_TR0':>7} {'S3_TR0':>7} {'dTR0':>6} {'S2_mean':>7} {'S3_mean':>7} {'S2_SR':>5} {'S3_SR':>5} {'dSR':>5} {'S2_slope':>8} {'S3_slope':>8}")
print("-" * 120)

for suite in SUITE_ORDER:
    s2_key = s3_key = None
    for (ms, cfg, s), st in all_stats.items():
        if s == suite and 'baseline' in cfg:
            if ms == S2_KEY:
                s2_key = (ms, cfg, s)
            elif ms == S3_KEY:
                s3_key = (ms, cfg, s)

    if s2_key and s3_key:
        s2 = all_stats[s2_key]
        s3 = all_stats[s3_key]
        sname = SUITE_SHORT.get(suite, suite)
        print(f"{sname:<16} {s2['tr_0']:>7.2f} {s3['tr_0']:>7.2f} {s3['tr_0']-s2['tr_0']:>+6.2f} "
              f"{s2['tr_mean']:>7.2f} {s3['tr_mean']:>7.2f} {s2['sr']:>5.0f} {s3['sr']:>5.0f} {s3['sr']-s2['sr']:>+5.0f} "
              f"{s2['tr_slope']:>8.4f} {s3['tr_slope']:>8.4f}")

# Also compare all S3 configs vs S2 baseline
s3_configs = sorted(set(c for (ms, c, s) in all_stats if ms == S3_KEY))
if len(s3_configs) > 1:
    print("\n" + "=" * 120)
    print("93f S3 ALL CONFIGS vs S2 BASELINE")
    print("=" * 120)
    print(f"{'S3_Config':<25} {'Suite':<16} {'S2_SR':>5} {'S3_SR':>5} {'dSR':>5} {'S2_TR0':>7} {'S3_TR0':>7} {'S3_slope':>8}")
    print("-" * 120)

    for cfg in s3_configs:
        for suite in SUITE_ORDER:
            s3_key = (S3_KEY, cfg, suite)
            s2_base = None
            for (ms, c, s) in all_stats:
                if ms == S2_KEY and s == suite and 'baseline' in c:
                    s2_base = (ms, c, s)
                    break
            if s3_key in all_stats and s2_base and s2_base in all_stats:
                s2 = all_stats[s2_base]
                s3 = all_stats[s3_key]
                sname = SUITE_SHORT.get(suite, suite)
                print(f"{cfg:<25} {sname:<16} {s2['sr']:>5.0f} {s3['sr']:>5.0f} {s3['sr']-s2['sr']:>+5.0f} "
                      f"{s2['tr_0']:>7.2f} {s3['tr_0']:>7.2f} {s3['tr_slope']:>8.4f}")
        print()

print(f"\nAll plots saved to {OUT}/")
print("Files: 18-23_*.png")
print(f"CSV: {csv_path}")
