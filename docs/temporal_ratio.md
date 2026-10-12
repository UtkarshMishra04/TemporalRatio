# Temporal Ratio and TR-adaptive guidance

## Temporal Ratio

The action head attends jointly to the DiT features of the 5 latent video frames (the current
frame 0 and 4 predicted future frames), the robot state and the noisy action tokens. For an
action-expert layer *l*, let *a_t* be the attention mass that the action queries put on the tokens
of latent frame *t* (summed over the frame's tokens and the action queries, averaged over heads, at
the first flow-matching step). Then

    TR_l = (a_1 + a_2 + ... + a_{T-1}) / a_0

TR is not normalized by token count: uniform attention over T frames gives TR = T − 1. High TR
means the action prediction leans on the model's own video rollout; low TR means it is anchored to
the observed frame. Implementations: `temporal_ratio.guidance.tr` (reference computation) and
`VideoVLAPytorch.sample_actions_adaptive_cfg` / `_compute_tr_from_storage` (computed during
inference from attention hooks).

`trvam infer` logs TR for `tr_log_layers` at every policy query (`report.json`), and
`VideoModelPolicyWrapper.infer(..., tr_layers=[...])` returns it as `temporal_ratio`.

## Guidance

At each flow-matching step of the action head, velocities from three conditionings are combined:

    v = v_c + w_lang (v_c − v_∅) + w_plan (v_plan − v_c)

- `v_c`: features from the video model conditioned on the task prompt (the normal policy),
- `v_∅`: features with an empty (zero) text embedding,
- `v_plan`: features from a longer video rollout of `plan_latent_frames` latent frames (24 = 93
  video frames), cropped to the 5 frames the head reads.

`formula: standard` uses `v_∅ + w_lang (v_c − v_∅)` instead of the extrapolation form.

| `guidance.type` | weights |
|---|---|
| `none` | no guidance |
| `flat` | constant `w_lang`, `w_plan` (`space: action` applies CFG to the final action chunk instead) |
| `tr_adaptive` | `w · s` with `s = clip(TR_k / TR_0 − 1, 0, s_max)` |

`TR_k` is the TR (layer `tr_layer`) at replanning step *k* and `TR_0` the episode's first TR
(`tr_baseline: per_episode`); `fixed` uses `tr_baseline_fixed` and `uncond_ratio` the TR of the
unconditional branch. Guidance thus switches on only when the policy relies on future frames more
than at the start of the episode, i.e. during planning-like phases.

`legacy_flat_floor: true` reproduces the original research code, which kept the flat base weights
whenever the adaptive scale was ~0 instead of switching guidance off. The default (`false`)
follows the method as written.

The math is in `temporal_ratio.guidance.scale` (unit-tested); the policy wrapper that runs the
three branches is `temporal_ratio.guidance.tr_adaptive_policy.AdaptiveActionCFGWrapper`.

## Analysis scripts

| script | purpose |
|---|---|
| `scripts/analysis/tr_probe.py` | TR across layers / denoising steps for LIBERO prompts and frames |
| `scripts/analysis/tr_probe_analyze.py` | statistics and plots of probe outputs |
| `scripts/analysis/tr_calibrate_baseline.py` | fixed TR baselines per checkpoint |
| `scripts/analysis/per_suite_tr_table.py` | TR trajectories of LIBERO rollouts per suite (`--results <eval output>`) |
