#!/usr/bin/env python3
"""Evaluate adaptive action-level CFG on LIBERO."""
import sys
import os

_W_BASE_LANG = float(os.environ.get("CFG_W_BASE_LANG", "5.0"))
_W_BASE_PLAN = float(os.environ.get("CFG_W_BASE_PLAN", "0.0"))
_PLAN_LAT = int(os.environ.get("CFG_PLAN_LAT", "24"))
_MAX_SCALE = float(os.environ.get("CFG_MAX_SCALE", "2.0"))
_TR_LAYER = int(os.environ.get("CFG_TR_LAYER", "2"))
_TR_LOG_LAYERS = [int(x) for x in os.environ.get("CFG_TR_LOG_LAYERS", "2,5,10,13,15").split(",")]
_TR_BASELINE_MODE = os.environ.get("CFG_TR_BASELINE_MODE", "per_episode")
_TR_BASELINE_FIXED = os.environ.get("CFG_TR_BASELINE_FIXED")
_TR_BASELINE_FIXED = float(_TR_BASELINE_FIXED) if _TR_BASELINE_FIXED else None
_CFG_FORMULA = os.environ.get("CFG_FORMULA", "extrapolation")
_LEGACY_FLAT_FLOOR = os.environ.get("CFG_LEGACY_FLAT_FLOOR", "0") == "1"
_DECODE_VIDEO = os.environ.get("TR_SKIP_VIDEO_DECODE", "0") != "1"

_ORIG_GET_POLICY = None


def _get_policy_with_adaptive_cfg(args, device, dummy=False):
    policy = _ORIG_GET_POLICY(args, device, dummy)
    if dummy:
        return policy

    from temporal_ratio.guidance.tr_adaptive_policy import AdaptiveActionCFGWrapper
    from temporal_ratio.policies import video_policy as _policy

    if isinstance(policy, _policy.VideoModelPolicyWrapper):
        policy = AdaptiveActionCFGWrapper(
            policy,
            w_base_lang=_W_BASE_LANG,
            w_base_plan=_W_BASE_PLAN,
            plan_latent_frames=_PLAN_LAT,
            max_scale=_MAX_SCALE,
            tr_layer=_TR_LAYER,
            tr_baseline_mode=_TR_BASELINE_MODE,
            tr_baseline_fixed=_TR_BASELINE_FIXED,
            cfg_formula=_CFG_FORMULA,
            tr_log_layers=_TR_LOG_LAYERS,
            legacy_flat_floor=_LEGACY_FLAT_FLOOR,
            decode_video=_DECODE_VIDEO,
        )
    return policy


sys.path.insert(0, os.path.dirname(__file__))
import eval_libero as eval_mod

_ORIG_GET_POLICY = eval_mod.get_policy
eval_mod.get_policy = _get_policy_with_adaptive_cfg

# Patch _run_episode to call reset_episode and save TR logs
_orig_run_episode = eval_mod._run_episode

import json
import pathlib
import numpy as np

def _run_episode_with_reset(args, policy, env, task_description,
                             task_suite_name, task_id, episode_idx, max_steps, video_out_path):
    if hasattr(policy, "reset_episode"):
        policy.reset_episode()

    result = _orig_run_episode(args, policy, env, task_description,
                                task_suite_name, task_id, episode_idx, max_steps, video_out_path)

    # Save full TR trajectory
    if hasattr(policy, "_tr_history") and policy._tr_history:
        tr_log_dir = pathlib.Path(video_out_path) / f"task_{task_id}" / "tr_logs"
        tr_log_dir.mkdir(parents=True, exist_ok=True)
        tr_log = {
            "episode_idx": episode_idx,
            "success": result.get("success", False),
            "suite": task_suite_name,
            "task_id": task_id,
            "tr_0": policy._tr_0,
            "n_replans": policy._replan_count,
            "trajectory": policy._tr_history,
        }
        with open(tr_log_dir / f"ep_{episode_idx}_tr.json", "w") as f:
            json.dump(tr_log, f)

    return result

eval_mod._run_episode = _run_episode_with_reset

# Override output path
_orig_vob = eval_mod._video_out_base
def _video_out_base_adaptive(args):
    base = _orig_vob(args)
    tag = (f"adaptive_wl{_W_BASE_LANG}_wp{_W_BASE_PLAN}_ms{_MAX_SCALE}"
           f"_L{_TR_LAYER}_base-{_TR_BASELINE_MODE}")
    return base.replace("/vid2world_", f"/{tag}/vid2world_")
eval_mod._video_out_base = _video_out_base_adaptive

if __name__ == "__main__":
    args = eval_mod.tyro.cli(eval_mod.Args)
    args.save_rollout_data = False
    sys.exit(eval_mod.main(args))
