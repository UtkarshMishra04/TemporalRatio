# Inference

An inference config (`configs/infer/`) names a checkpoint, the robot/dataset config it was trained
with and the inference settings ([configs.md](configs.md#fields)). The same loader
(`temporal_ratio.inference.load_policy`) is used by both entry points:

| entry point | use |
|---|---|
| `trvam serve <cfg>` | websocket policy server for a real robot / external simulator |
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

The action head is conditioned on the features' noise level in the variable it was trained with:
the fraction `u` in [0, 1) that the DiT's training scheduler maps to that noise level (e.g.
`u` = 0.336 for sigma 0.908, i.e. 6 video steps with features after 2).

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

## Serving

```bash
trvam serve configs/infer/abc130k/yam_pp24.yaml --set checkpoint=<ckpt> --port 8000
```

The server speaks the openpi websocket protocol: connect with
`openpi_client.websocket_client_policy.WebsocketClientPolicy(host, port)` and call
`client.infer(obs)` with the observation dict above. The response holds `actions`,
`policy_timing`, `server_timing`, `temporal_ratio` (with `tr_log_layers`; string keys) and, with
guidance, `adaptive_cfg_info`.

**Predicted video.** Add `"generate_video": True` (or `False`) to an observation to override the
config's `generate_video` for that query. The response then also carries `future_video`
(T x H x W x 3 uint8, the 17 predicted frames with the cameras tiled left to right; frame 0 is the
decoded conditioning frame); it is absent otherwise. `--save-video DIR` makes the server write
every returned video to `DIR/query_<n>.mp4` (10 fps); the responses are unchanged. Video
generation runs the full denoising schedule with video CFG and a VAE decode, roughly doubling the
query latency (see the timings above).

**Client check.** `scripts/check_server.py` sends observations built from image files (one per
camera, in the order of the table above) or random images, validates the returned action chunks
(finite, 2-D) and exits non-zero otherwise; `--video-dir DIR` requests the predicted video and
saves `query_<n>.mp4` with a contact sheet `query_<n>.png` (frames 0, 4, 8, 12, 16). The robot is
taken from the server's `policy_config`.

```bash
python scripts/check_server.py --port 8000 --random --queries 3
python scripts/check_server.py --port 8000 --images top.png left_wrist.png right_wrist.png \
    --state 0 0 0 0 0 0 0 0 0 0 0 0 0 0 --prompt "fold the t-shirt" --video-dir results/check_server
```
