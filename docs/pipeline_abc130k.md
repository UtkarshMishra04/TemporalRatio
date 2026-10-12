# Pipeline: ABC-130k T-shirt folding, pen caps, bottle caps

End-to-end training and evaluation on three ABC-130k tasks: `fold_and_stack_the_t_shirts`
(500 sampled episodes), `open_the_pen_caps` and `unscrew_the_bottle_caps` — the `fold_pen_bottle`
task group of `src/temporal_ratio/data/abc130k.py`. Folding and bottle caps are part of the released
YAM checkpoints' training set; pen caps is new to them.

```bash
export HF_LEROBOT_HOME=/path/to/abc130k_lerobot     # where the converted datasets live
```

## 1. Data

Convert the episodes (train and validation splits, about 37 GB in total; see
[data.md](data.md#abc-130k-bimanual-yam) for the commands and details). Normalization statistics
for training from the base model ship in
`assets/videoVLA_abc130k_fold_pen_bottle_short/`; fine-tuning keeps the checkpoint's own.

## 2. Evaluate a checkpoint

```bash
trvam infer configs/infer/abc130k/fold_pen_bottle.yaml --set checkpoint=<ckpt step dir>
```

Evaluates open-loop on `episodes` of the three validation splits (`offline` section of the config):
every `replan_every` frames the policy predicts a 50-step action chunk from the recorded
observation, which is compared with the recorded actions. `results/infer/fold_pen_bottle/report.json`
holds per-query action MSE / L1, the Temporal Ratio of `tr_log_layers` and latencies, with
per-dataset summaries, also in units of each validation split's action standard deviation
(`*_std`). `configs/infer/abc130k/fold_pen_bottle_heldout.yaml` covers whole validation episodes
instead of their first 200 frames: one query every 150 frames of up to 20 episodes per split. Add
`--set generate_video=true` to also write the predicted future video
next to the recorded one, and `--set offline.episodes=[0,1,...]` / `--set offline.max_steps=...` to
change the budget.

## 3. Train

Fine-tune a released checkpoint (its DiT becomes the base of new LoRA adapters, its action head the
initialization, its normalization statistics are kept):

```bash
TR_FINETUNE_FROM=<ckpt step dir> trvam train configs/train/abc130k/finetune_from_checkpoint.yaml --gpus 1
```

To fine-tune on a single task, select its task group (`folding`, `pen_caps`, or `assembly` for
bottle caps) and evaluate on its validation split with
`trvam infer ... --dataset abc130k/<task>_val`:

```bash
TR_FINETUNE_FROM=<ckpt step dir> trvam train configs/train/abc130k/finetune_from_checkpoint.yaml --gpus 1 \
    --set data.action_dataset=videoVLA_abc130k_pen_caps_short --set data.video_dataset=abc130k_pen_caps
```

One pass over the non-overlapping 50-step action chunks takes frames / 50 / action batch size
steps: 11,163 for the three tasks together (4.47 M frames at action batch 8), and 3,633 / 5,051 /
2,480 for folding / pen caps / bottle caps alone (`--set stages.0.steps=...`). Samples are drawn
with replacement, balanced across tasks. Add `--set stages.0.keep_every=<steps of one epoch>` to keep
every epoch's checkpoint. To train for more epochs later, repeat the command with a larger
`stages.0.steps`; the run resumes from its latest checkpoint
(see [training.md](training.md#continuing-a-run)).

To train from the pretrained Cosmos-Predict2.5 2B model instead, use the multi-task recipe:

```bash
trvam train configs/train/abc130k/fold_pen_bottle_joint_lora_r32.yaml --gpus 8 --set stages.0.steps=200000
```

Both use the dual-resolution trainer (separate video and action batches, 17-frame clips at frame
skip 3, 50-step action chunks). `--gpus N` runs on one node; for several nodes set
`distributed.{nnodes,node_rank,master_addr}` instead. Checkpoints are written to
`checkpoints/videoVLA_abc130k_fold_pen_bottle_short/<run>/<step>/` every `save_interval` steps; use
`logging.wandb=true` after `wandb login`. Useful overrides: `stages.0.steps`, `stages.0.action_lr` /
`video_lr`, `optim.video_batch_size` / `optim.action_batch_size`,
`adaptation.lora.{rank,alpha,target_modules}`, `adaptation.video=full`,
`stages.0.action_first_frame_drop_prob` (see [configs.md](configs.md#stage-modes)), `data.lerobot_home`.

## 4. Evaluate and deploy the result

```bash
trvam infer configs/infer/abc130k/fold_pen_bottle.yaml --set checkpoint=checkpoints/.../<step>
trvam serve configs/infer/abc130k/fold_pen_bottle.yaml --set checkpoint=checkpoints/.../<step> --port 8000
```

The server expects the YAM observation keys of [inference.md](inference.md#observations-and-outputs)
and returns 50 x 14 action chunks (absolute joint and gripper targets of both arms).
