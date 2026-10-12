# Data

Training uses two data streams:

- **action data**: a named robot/dataset config in `src/temporal_ratio/config/registry.py`
  (`data.action_dataset`) — a LeRobot dataset, its key mapping, the robot input/output transforms,
  action / observation horizons, frame skip and the video setup;
- **video data**: a mixture in `src/temporal_ratio/data/video_datasets.py` (`data.video_dataset`)
  of LeRobot or local video datasets, used for the video (DiT) loss.

LeRobot datasets are read from `$HF_LEROBOT_HOME/<repo id>` (default
`~/.cache/huggingface/lerobot`) and downloaded from the Hub on first use if missing.

## LIBERO

| | |
|---|---|
| action data | `videoVLA_libero_short`: [`physical-intelligence/libero`](https://huggingface.co/datasets/physical-intelligence/libero) (33 GB), agent-view + wrist camera, 8-D state, 7-D actions, 17-step chunks |
| video data | `libero`: the same demonstrations (LIBERO-Spatial/Object/Goal/10) + LIBERO-90 from [`IPEC-COMMUNITY/libero_90_no_noops_lerobot`](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_90_no_noops_lerobot) (3.4 GB), 50/50 |
| video setup | `libero_flat_short`: cameras tiled to 224x448 |
| norm stats | shipped in `assets/videoVLA_libero_short/physical-intelligence/libero/norm_stats.json` |

The paper's LIBERO runs additionally mixed 10% failed rollouts into the video data; those
videos are not part of this release. `video_dataset: libero_with_failures` adds them back from a
local directory `$TR_LIBERO_FAILURES_DIR` with `videos/*.mp4` and `metas/*.txt` (one prompt per
video).

## ABC-130k (bimanual YAM)

[ABC-130k](https://huggingface.co/datasets/XDOF/ABC-130k) contains about 130k teleoperated
episodes from two 6-DoF YAM arms with parallel grippers, a top camera and two wrist cameras,
stored as MCAP files. The released YAM checkpoints were trained on 24 tasks in a LeRobot layout;
22 of them are part of ABC-130k (`src/temporal_ratio/data/abc130k.py` lists them and their
original names).

Convert episodes with

```bash
# 50 episodes of one task from the Hub -> $HF_LEROBOT_HOME/abc130k/fold_and_stack_the_t_shirts
python scripts/data/convert_abc130k_to_lerobot.py --src hf:fold_and_stack_the_t_shirts --max-episodes 50

# a whole task, deleting each raw episode once converted (rerun with --resume after interruptions)
python scripts/data/convert_abc130k_to_lerobot.py --src hf:load_the_cups_into_the_dish_rack \
    --workers 24 --delete-raw
```

`--src` also accepts local directories containing `*/episode.mcap`. The converter has its own
requirements (`requirements/convert.txt`: `mcap`, `av`, `pyarrow`, ...; not `lerobot`). It resamples all streams to a
uniform 30 Hz grid (nearest timestamp), resizes the cameras to 168x224 and writes LeRobot v2.1
datasets readable by the training stack (`datasets<3`):

- `observation.state`, `action`: 14-D, `[left joints 1-6 (rad), left gripper, right joints 1-6,
  right gripper]`, gripper as normalized aperture (0 = closed, 1 = open); `action` is the
  commanded (teleoperation) position;
- `observation.images.{left_image, left_wrist_left, right_wrist_left}`: top camera (left eye on
  ZED-X stations), left wrist, right wrist; AV1 video;
- the episode instruction as task prompt (`--prompts-json` supplies paraphrases).

Differences to the data the released checkpoints were trained on: ZED-X frames (16:10) are
center-cropped to 4:3; 60 Hz camera streams are decimated to 30 Hz; images and robot states are
paired by timestamp (`--camera-offset-ms 104` pairs by frame index instead, which matches the
~104 ms offset of the 30 Hz RealSense stations); the original training set used ~1,000
paraphrased prompts and clipped episodes (`--max-frames` limits episode length). Per-dimension
state/action statistics of converted episodes are within 0.25 standard deviations of the
checkpoints' normalization statistics.

Converting all 22 tasks (about 22.8k episodes, 2.9 TB of MCAP) takes roughly 0.5-0.9 hours per
1,000 episodes with 24 workers and produces about 20-25 GB per 1,000 episodes; with
`--delete-raw` the raw data only occupies disk transiently, and the download is usually the
bottleneck.

| named config | tasks |
|---|---|
| `videoVLA_abc130k_fold_pen_bottle_short` | `fold_and_stack_the_t_shirts`, `open_the_pen_caps`, `unscrew_the_bottle_caps` (video: `abc130k_fold_pen_bottle`) |
| `videoVLA_abc130k_pen_caps_short` | `open_the_pen_caps` (video: `abc130k_pen_caps`); single-task folding and bottle caps are `..._folding_short` / `..._assembly_short` |
| `videoVLA_abc130k_pp24_short` | the 22 tasks shared with the checkpoints' training set (video: `abc130k_pp24`) |
| `videoVLA_abc130k_{folding,garment,assembly,pickplace}_short` | one skill group (video: `abc130k_<skill>`) |

The `fold_pen_bottle` data uses all episodes of the pen-cap and bottle-cap tasks and a fixed
random subset of 500 T-shirt-folding episodes:

```bash
convert=scripts/data/convert_abc130k_to_lerobot.py
python $convert --src hf:fold_and_stack_the_t_shirts --max-episodes 500 --select random --seed 0 --workers 24 --delete-raw
python $convert --src hf:open_the_pen_caps --workers 24 --delete-raw
python $convert --src hf:unscrew_the_bottle_caps --workers 24 --delete-raw
for task in fold_and_stack_the_t_shirts open_the_pen_caps unscrew_the_bottle_caps; do   # validation split
  python $convert --src hf:val/$task --repo-id abc130k/${task}_val --workers 24 --delete-raw
done
```

| task | train episodes (of available) | val episodes | MCAP download |
|---|---|---|---|
| `fold_and_stack_the_t_shirts` | 500 (of 11,009) | 82 | 62 GB (+14 GB val) |
| `open_the_pen_caps` | 516 | 7 | 116 GB |
| `unscrew_the_bottle_caps` | 294 | 5 | 43 GB |

Normalization statistics for `videoVLA_abc130k_fold_pen_bottle_short` ship in `assets/`.
`open_the_pen_caps` is not part of the released checkpoints' training set. Validation episodes
(`abc130k/<task>_val`) can be used for open-loop evaluation, e.g.
`trvam infer configs/infer/abc130k/yam_pp24.yaml --set checkpoint=<ckpt> --dataset abc130k/open_the_pen_caps_val`.

All use 50-step action chunks, observations sub-sampled with frame skip 3, and the
`abc130k_flat_short` video setup (cameras padded to 176x224 and tiled to 176x672). Compute the
normalization statistics after converting:

```bash
python scripts/data/compute_norm_stats.py --config-name videoVLA_abc130k_pp24_short --fast
```

When fine-tuning a released checkpoint, reuse its statistics instead with
`NORM_STATS_PATH=<ckpt>/assets/<...>/norm_stats.json`.

## Adding a robot

1. Write input/output transforms mapping your observation keys to the model's image / state keys
   (`src/temporal_ratio/policies/libero_policy.py` and `yam_policy.py` are templates).
2. Add a `VideoSetup` in `src/temporal_ratio/models/video_model.py`: Cosmos experiment, size of the
   tiled frame (multiples of 16) and the camera order.
3. Register a `TrainConfig` in `src/temporal_ratio/config/registry.py` (LeRobot repo id, key
   mapping, horizons, frame skip, `video_setup`) and a video mixture in
   `src/temporal_ratio/data/video_datasets.py`.
4. Compute normalization statistics and point `data.action_dataset` / `data.video_dataset` of a
   training config at the new names.
