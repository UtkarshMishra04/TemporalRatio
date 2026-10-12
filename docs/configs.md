# Experiment configs

All training and inference is driven by YAML experiment configs under `configs/`. A config is
validated against the dataclasses in `src/temporal_ratio/config/schema.py` (unknown keys, wrong
types and inconsistent settings are errors) and translated into trainer / evaluator invocations by
`src/temporal_ratio/config/translate.py`.

```bash
trvam show  <config.yaml> [--set key=value ...]   # resolved config + the exact commands it runs
trvam train <train.yaml>  [--set ...] [--dry-run] [--only STAGE] [--from STAGE]
trvam infer <infer.yaml>  [--set ...] ...         # open-loop evaluation on recorded episodes
trvam serve <infer.yaml>  [--set ...] [--port N]  # websocket policy server
trvam eval-libero <eval.yaml> [--set ...]         # closed-loop LIBERO
```

## Composition

- `_base_: <path>` (relative to the file) inherits another config; mappings are merged
  recursively and lists are replaced. Nested mappings may have their own `_base_`
  (e.g. `inference: {_base_: ../../infer/libero/tr_adaptive.yaml}` in eval configs).
- `${VAR}` and `${VAR:-default}` are replaced by environment variables; a leading `~` is expanded.
- `--set a.b.c=value` overrides any field after inheritance (`stages.0.steps=200`,
  `adaptation.lora.target_modules=[q_proj,v_proj]`). Values are parsed as YAML.

Files whose name starts with `_` (e.g. `train/libero/_libero.yaml`) are shared bases, not runnable
presets.

## Training configs

```yaml
_base_: ../../base/cosmos2b_gemma300m.yaml
name: libero_two_stage_lora_r32
paradigm: two_stage            # joint | action_only | two_stage | three_stage | custom
model:
  base_video_checkpoint: base  # `base` = Cosmos-Predict2.5-2B, or a video_model_full.pt
  action_head_init: null       # checkpoint dir with model.safetensors to initialize the head
  feature_layer: 19            # DiT block whose output feeds the action head
  action_horizon: 17           # action chunk length
  observation_horizon: 17      # video frames seen by the action head (5 latent frames)
adaptation:
  video: lora                  # lora | full | frozen
  lora: {rank: 32, alpha: 32, target_modules: [q_proj, k_proj, v_proj, output_proj, mlp.layer1, mlp.layer2]}
data:
  action_dataset: videoVLA_libero_short   # named robot/dataset config (config/registry.py)
  video_dataset: libero                   # video mixture for the video loss (data/video_datasets.py)
  video_num_frames: 17                    # != 17 selects the dual-resolution trainer
stages:
  - {name: s1_video,  mode: video_only,   steps: 16000, video_lr: 1.0e-4, action_lr: 1.0e-6}
  - {name: s2_action, mode: frozen_video, steps: 50000, video_lr: 0.0,    action_lr: 1.0e-4,
     video_loss_weight: 0.0, init_video_from: previous_stage}
```

### Adaptation of the video DiT

| `adaptation.video` | effect |
|---|---|
| `lora` | PEFT LoRA adapters on `adaptation.lora.target_modules` of every DiT block; base weights frozen. Targets: `q_proj`, `k_proj`, `v_proj`, `output_proj` (self- and cross-attention) and `mlp.layer1`, `mlp.layer2`. |
| `full` | all DiT parameters are trained. |
| `frozen` | no DiT parameters are trained (every stage must be `frozen_video`). |

Checkpoints always store the merged DiT (`video_model_full.pt`); LoRA runs also store
`lora_adapters/`.

### Stage modes

| `mode` | video loss | action loss | gradients into the DiT |
|---|---|---|---|
| `joint_full` | yes (`video_loss_weight`) | yes | video and action loss |
| `joint_detach` | yes | yes | video loss only (features detached) |
| `video_only` | yes | no (head frozen) | video loss |
| `frozen_video` | no | yes | none (DiT frozen; `video_lr` is ignored) |

`init_video_from` is `base` (the model's `base_video_checkpoint`), `previous_stage` (the latest
checkpoint of the preceding stage) or a path. `init_action_from` is `null`, `previous_stage` or a
checkpoint directory. Checkpoints go to `<output_dir>/<action_dataset>/<run>/<step>/`, where `<run>`
is the experiment `name` (`<name>_<stage name>` in multi-stage recipes), next to an
`experiment.yaml` with the resolved config.

`action_first_frame_drop_prob` (per stage, default 0) hides the first latent frame, i.e. the clean
conditioning frame, from the action head for that fraction of action samples, so that the head
learns to use the DiT features of the predicted frames. The DiT still sees the frame, and inference
is unchanged.

`keep_every` (per stage) also saves at every multiple of that many steps and never deletes those
checkpoints. Runs resume from their latest checkpoint by default (`resume: true`); `overwrite: true`
starts over. See [training.md](training.md#continuing-a-run) for training a run for more epochs.

`paradigm` only constrains the stage sequence: `joint` = `[joint_full]`, `action_only` =
`[frozen_video]`, `two_stage` = `[video_only, frozen_video]`, `three_stage` =
`[video_only, joint_full, frozen_video]`; `custom` allows any sequence.

### Trainers

- **standard** (`temporal_ratio.training.train`): each batch provides 17-frame clips used for the
  video loss and the action loss.
- **dual_resolution** (`temporal_ratio.training.train_dual_resolution`): separate video batches
  (`video_num_frames`, e.g. 93-frame clips = 24 latent frames, `optim.video_batch_size`) and action
  batches (17 frames, `optim.action_batch_size`); `video_aux_every_k` interleaves an auxiliary
  17-frame video loss.

`trainer: auto` picks dual-resolution when `data.video_num_frames != 17`.

### Optimization and logging

`optim`: per-GPU `batch_size` (standard trainer), `video_batch_size` / `action_batch_size`
(dual-resolution), `warmup_ratio`, cosine schedule down to `min_lr_ratio`, AdamW (`weight_decay`,
`beta2`), `gradient_clip_norm`, EMA (`ema_decay`, `ema_type`), `gradient_checkpointing` (per-stage
override available). Learning rates are per stage (`video_lr`, `action_lr`).

`logging.wandb` enables Weights & Biases (credentials from `wandb login`; entity from
`$WANDB_ENTITY`). `distributed` sets the `torchrun` topology (`nnodes`, `nproc_per_node`,
`master_addr`, `master_port`, `node_rank`; run the same command on every node with its
`node_rank`). Setting `$CHECKPOINT_SYNC_URI` uploads every checkpoint with `s5cmd sync`.

## Inference configs

```yaml
checkpoint: ${TR_CHECKPOINT:-checkpoints/...}   # step dir, or hf://<owner>/<repo>/<dir>
policy_config: videoVLA_libero_short            # named config: transforms, dims, video setup
video_weights: auto           # auto | full (video_model_full.pt) | lora (base + lora_adapters/)
feature_layer: null           # default: the checkpoint's own (metadata.pt), else 19
video_denoising_steps: 6      # length of the video denoising schedule
feature_extraction_step: 2    # the action head reads DiT features after this many steps
action_denoising_steps: 10    # flow-matching steps of the action head
generate_video: false         # true: full schedule with video CFG + decode the predicted frames
video_guidance: 7.0           # video CFG weight (only with generate_video)
replan_every: 10              # LIBERO / offline: query the policy every N steps
tr_log_layers: [2, 5, 10, 13, 15]
guidance:
  type: none                  # none | flat | tr_adaptive  (see docs/temporal_ratio.md)
```

`generate_video: false` stops the video denoising once the features are available and skips video
CFG, which is what the policy needs for control. `true` additionally finishes the schedule and
decodes the future frames (`future_video` in the policy output); because CFG changes the
denoising trajectory, the action head then sees slightly different features.

## LIBERO evaluation configs

```yaml
inference: {_base_: ../../infer/libero/tr_adaptive.yaml}
eval:
  split: id                  # id | ood | libero_90 | all
  num_trials_per_task: 10
  seeds: [7]
  num_steps_wait: 10
  output_dir: results/libero/id
  save_rollout_data: false
```
