# Inference

An inference config (`configs/infer/`) names a checkpoint, the robot/dataset config it was trained
with and the inference settings ([configs.md](configs.md#inference-configs)). The same loader
(`temporal_ratio.inference.load_policy`) is used by every entry point:

| entry point | use |
|---|---|
| `trvam infer <cfg>` | open-loop evaluation on recorded LeRobot episodes |
| `trvam serve <cfg>` | websocket policy server for a real robot / external simulator |
| `trvam eval-libero <eval cfg>` | closed-loop LIBERO evaluation |
| `load_policy(exp)` | Python API |

## One policy query

1. The robot's input transform maps the observation to model inputs; states and actions are
   normalized with the checkpoint's statistics (`assets/**/norm_stats.json`, or `norm_stats`).
2. The cameras are tiled into one frame (`libero_flat_short`: 224x448, `abc130k_flat_short`:
   176x672) and the Cosmos-Predict2.5 Video2World model denoises a 17-frame (5 latent frames) future
   conditioned on that frame and the prompt.
3. After `feature_extraction_step` of `video_denoising_steps` steps, the DiT block-`feature_layer`
   features of the 5 latent frames condition the action head, which integrates
   `action_denoising_steps` flow-matching steps into an action chunk.

With `generate_video: false` (default) the video model stops at step 3 and skips video CFG. With
`generate_video: true` it runs the full schedule with video CFG (`video_guidance`) and the VAE
decodes the predicted frames; the action head then reads features from that trajectory.

Measured on one GB300 (bimanual YAM checkpoints, 6 video steps, features after 2, TR logging on):
loading takes 60-80 s; a warm query takes about 1.5 s action-only and 3 s with video generation
(about 0.8 s / 1.4 s without TR logging); peak memory is about 27 GB.

## Observations and outputs

| robot | observation keys | action chunk |
|---|---|---|
| LIBERO (`videoVLA_libero_short`) | `observation/image`, `observation/wrist_image` (HxWx3 uint8), `observation/state` (8), `prompt` | 17 x 7 |
| YAM / ABC-130k (`videoVLA_abc130k_*`) | `observation/left_image` (top), `observation/left_wrist_left`, `observation/right_wrist_left`, `observation/state` (14), `prompt` | 50 x 14 |

`infer` returns `actions` (un-normalized), `future_video` (T x H x W x 3 uint8, or `None`),
`features`, `policy_timing` and, when `tr_layers` / `tr_log_layers` are given, `temporal_ratio`
(`{layer: TR}`).

```python
from temporal_ratio.config import loader
from temporal_ratio.inference import load_policy

exp = loader.load_inference("configs/infer/abc130k/yam_pp24.yaml", ["checkpoint=/path/to/ckpt"])
policy = load_policy(exp)
out = policy.infer(obs, generate_video=True, tr_layers=[2, 10])
out["actions"].shape, out["future_video"].shape, out["temporal_ratio"]
```

## Open-loop evaluation

```bash
trvam infer configs/infer/libero/no_guidance.yaml --set checkpoint=<ckpt> \
    --dataset physical-intelligence/libero --episodes 0 12 --max-steps 20 --out results/infer/libero
```

Every `replan_every` frames the recorded observation and prompt are fed to the policy and the
predicted chunk is compared with the recorded actions. `report.json` contains per-query action
MSE / L1, TR per layer and latency, plus per-episode summaries; the errors are also given in units
of the evaluated dataset's per-dimension action standard deviation (`action_rmse_std`,
`action_mae_std`, `first_action_mae_std`, from the dataset's statistics; with `offline.first_k`,
also `first_k_rmse_std` / `first_k_mae_std` over the first `first_k` actions of each chunk); `episode_<e>.npz` holds predicted
and recorded chunks; with `generate_video=true`, `episode_<e>_step_<t>.mp4` shows the predicted
future (top) above the recorded one (bottom). Guided configs (`tr_adaptive.yaml`, `flat.yaml`)
work the same way.

## Serving

```bash
trvam serve configs/infer/abc130k/yam_pp24.yaml --set checkpoint=<ckpt> --port 8000
```

The server speaks the openpi websocket protocol: connect with
`openpi_client.websocket_client_policy.WebsocketClientPolicy(host, port)` and call
`client.infer(obs)` with the observation dict above.

## Closed-loop LIBERO

Requires the LIBERO simulator ([install.md](install.md#libero-simulation-closed-loop-evaluation-only)).

```bash
trvam eval-libero configs/eval/libero/id.yaml  --set inference.checkpoint=<ckpt>
trvam eval-libero configs/eval/libero/ood.yaml --set inference.checkpoint=<ckpt> \
    --set inference.guidance.type=none
```

The evaluator (`scripts/eval/eval_libero.py`, or `eval_libero_guided.py` with guidance) runs
`num_trials_per_task` episodes per task and seed, writes one mp4 per rollout
(`rollout_success_*.mp4` / `rollout_failure_*.mp4`) and, for guided runs, per-episode TR traces
(`tr_logs/`). It can be parallelized over GPUs with `torchrun --nproc_per_node=N`.
