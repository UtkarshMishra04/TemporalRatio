# Training

```bash
trvam show  configs/train/libero/joint_lora_r32.yaml        # inspect: resolved config + commands
trvam train configs/train/libero/joint_lora_r32.yaml --dry-run
trvam train configs/train/libero/joint_lora_r32.yaml
```

`trvam train` runs one `torchrun` command per stage (`python -m torch.distributed.run`, same
interpreter) with the trainer arguments generated from the config, and resolves
`previous_stage` initializations from the latest checkpoint of the preceding stage.
`--only <stage>` runs a single stage, `--from <stage>` resumes a recipe; runs resume from their
latest checkpoint by default (`resume: true`).

Configs default to 2 nodes x 8 GPUs (`distributed`), as used for the paper. For a single node:

```bash
trvam train <cfg> --set distributed.nnodes=1 --set distributed.nproc_per_node=8
```

For multiple nodes, run the same command on every node with `--set distributed.node_rank=<i>`
and `--set distributed.master_addr=<node 0>`.

## What gets trained

The video model (Cosmos-Predict2.5 2B DiT, initialized from `model.base_video_checkpoint`) and the
action head (Gemma-300M expert, randomly initialized or from `model.action_head_init`) form a
joint model. The video loss is the flow-matching loss of the DiT on video clips (conditioned on the
first frame and the prompt); the action loss is the flow-matching loss of the action head, which
reads the DiT's block-`feature_layer` features of the current frame plus 4 predicted latent frames
at a sampled noise level. Stage `mode` selects which losses run and where gradients flow (see
[configs.md](configs.md#stage-modes)); the VAE and the text encoder are always frozen.

Text prompts are encoded with Cosmos-Reason1-7B; during training the conditioner drops the text
embedding with probability 0.2 (needed for classifier-free guidance at inference).

## Checkpoints

Every `save_interval` steps a stage writes `<output_dir>/<action_dataset>/<run>/<step>/`:

| file | content |
|---|---|
| `model.safetensors` | action head loaded for inference (same weights as `action_model.safetensors`) |
| `action_model_ema.safetensors` | EMA of the action head |
| `video_model_full.pt` | DiT weights (LoRA merged) |
| `lora_adapters/` | LoRA adapters (LoRA runs) |
| `ema.pt`, `optimizer.pt` | training state for resuming |
| `metadata.pt` | step and the trainer config (video setup, feature layer, ...) |
| `assets/<repo id>/norm_stats.json` | normalization statistics used for training |

The 5 most recent checkpoints of a run are kept. With `stages.N.keep_every=<steps>` the stage also
saves at every multiple of that many steps (e.g. one epoch) and never deletes those checkpoints.
`$CHECKPOINT_SYNC_URI` (e.g. an S3 prefix) uploads each checkpoint in the background with `s5cmd`.

## Continuing a run

Every checkpoint holds the full training state, and runs resume by default (`resume: true`):
repeating a command continues from the latest checkpoint of its run directory, restoring the LoRA
adapters, the action head, the EMA, the optimizer state and the step. To train longer, repeat the
command, with the same configuration and base checkpoint, and a larger step budget:

```bash
# one epoch of 3,633 steps, keeping the checkpoint of every epoch
trvam train <config> <overrides> --set stages.0.steps=3633 --set stages.0.keep_every=3633
# later: continue the same run to three epochs
trvam train <config> <overrides> --set stages.0.steps=10899 --set stages.0.keep_every=3633
```

- A new step budget starts a new learning-rate cycle at the resumed step: warmup and cosine decay
  over the added steps, so every continuation ends annealed. Resuming with the same budget (e.g.
  after a crash) stays in the current cycle, which `metadata.pt` records as `lr_cycle_start`.
- Data order and noise are reseeded with `seed + step`, so continued training draws new samples.
- A run whose directory already has checkpoints is never overwritten silently: `overwrite=true`
  deletes them and starts over.

## Fine-tuning a released checkpoint

```bash
TR_FINETUNE_FROM=<ckpt step dir> trvam train configs/train/libero/finetune_from_checkpoint.yaml
TR_FINETUNE_FROM=<ckpt step dir> NORM_STATS_PATH=<ckpt>/assets/.../norm_stats.json \
    trvam train configs/train/abc130k/finetune_from_checkpoint.yaml
```

The checkpoint's merged DiT becomes the base for new LoRA adapters and its action head is the
initialization. Reusing the checkpoint's normalization statistics keeps the action head's input
and output scaling unchanged.

## Resources

With LoRA r32, one GPU, video batch 1 and action batch 2 (ABC-130k, 17-frame clips), training
peaks at about 72 GB of GPU memory, including the two Reason1 text encoder copies (inference
pipeline and training conditioner). The first step takes several minutes (data loader start-up,
text encoder loading); later steps take about a second at this batch size. Lower
`optim.*batch_size` or enable `optim.gradient_checkpointing` on smaller GPUs.
