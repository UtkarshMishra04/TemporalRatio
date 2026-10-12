#!/usr/bin/env python3
"""Evaluate action-level CFG on LIBERO ID+OOD.

Wraps VideoModelPolicyWrapper with ActionCFGWrapper and runs standard eval.
Tests multiple guidance weights.

Usage:
    torchrun --standalone --nproc_per_node=4 scripts/eval_action_cfg.py \
        --checkpoint_step 60000 --num_trials_per_task 5 --eval_all
"""

# Reuse the existing eval infrastructure
import sys
import os

# Monkey-patch the policy loading to add CFG wrapper
_ORIG_GET_POLICY = None
_W_LANG = float(os.environ.get("CFG_W_LANG", "1.5"))
_W_PLAN = float(os.environ.get("CFG_W_PLAN", "0.0"))
_PLAN_LAT = int(os.environ.get("CFG_PLAN_LAT", "24"))


def _get_policy_with_cfg(args, device, dummy=False):
    """Load policy and wrap with ActionCFGWrapper."""
    policy = _ORIG_GET_POLICY(args, device, dummy)
    if dummy:
        return policy

    from temporal_ratio.guidance.action_cfg_policy import ActionCFGWrapper
    from temporal_ratio.policies import video_policy as _policy

    # Only wrap if it's a VideoModelPolicyWrapper
    if isinstance(policy, _policy.VideoModelPolicyWrapper):
        policy = ActionCFGWrapper(
            policy,
            guidance_weight=_W_LANG,
            plan_weight=_W_PLAN,
            plan_latent_frames=_PLAN_LAT,
        )
    return policy


# Import and patch the eval script
sys.path.insert(0, os.path.dirname(__file__))
import eval_libero as eval_mod

_ORIG_GET_POLICY = eval_mod.get_policy
eval_mod.get_policy = _get_policy_with_cfg

# Override output path to include CFG params
_orig_video_out_base = eval_mod._video_out_base
def _video_out_base_cfg(args):
    base = _orig_video_out_base(args)
    return base.replace("/vid2world_", f"/cfg_wl{_W_LANG}_wp{_W_PLAN}/vid2world_")
eval_mod._video_out_base = _video_out_base_cfg

if __name__ == "__main__":
    args = eval_mod.tyro.cli(eval_mod.Args)
    args.save_rollout_data = False  # Never save npz for CFG eval
    sys.exit(eval_mod.main(args))
