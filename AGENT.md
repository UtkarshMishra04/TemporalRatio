# AGENT.md

Instructions for coding agents (Claude Code, Codex, Cursor, ...) that set up and run this
repository for a user. Work through the steps in order. Each step ends with a **Check**; do not go
on until it passes. Human-oriented documentation: [README.md](README.md) and [docs/](docs/).

## Rules

- Never print, log, commit or write credentials (Hugging Face, W&B, GitHub tokens) into this
  repository. Hugging Face access goes through `hf auth login`, which the user runs, or an
  `HF_TOKEN` the user has exported. The test suite and CI scan for secrets.
- The GPU may be shared. Run `nvidia-smi` before starting a GPU job, and never stop processes you
  did not start. Inference needs about 30 GB of GPU memory; training needs 70–160 GB depending on
  batch sizes (step 5b).
- Start long jobs detached and poll their logs:
  `setsid nohup <command> > <log> 2>&1 < /dev/null &`.
- After changing code or configs, run `pytest -q`. The golden tests pin the trainer arguments of
  every preset (`tests/launcher_argv.json`); update them only for intended changes.
- Ask the user before large downloads or data conversions (tens of GB), before deleting
  checkpoints, and before pushing.

## 1. Environment

Requirements: Linux (x86_64 or aarch64), an NVIDIA GPU with a driver for CUDA 13 (for CUDA 12.8 set
`TORCH_INDEX=https://download.pytorch.org/whl/cu128`), [uv](https://docs.astral.sh/uv/), and about
100 GB of disk for the environment, models and checkpoints, plus the data.

```bash
git clone https://github.com/UtkarshMishra04/TemporalRatio.git && cd TemporalRatio
bash scripts/setup/install.sh            # creates .venv (Python 3.11); see docs/install.md
source .venv/bin/activate
```

**Check:**

```bash
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())"
trvam --help        # subcommands: show, train, infer, serve, eval-libero, download
pytest -q           # all tests pass
```

Without a GPU (configs, tests, dry runs only): `uv venv --python 3.11 && uv pip install -e ".[test]"`.

## 2. Pretrained models

The first inference or training run downloads `nvidia/Cosmos-Predict2.5-2B` (DiT and VAE, 4.6 GB)
and `nvidia/Cosmos-Reason1-7B` (text encoder, 16.6 GB) into the Hugging Face cache. The user must
accept both licenses on the Hub. If a download fails with HTTP 401 or 403, ask the user to accept
the licenses and run `hf auth login`. Once downloaded, `export HF_HUB_OFFLINE=1` makes runs
independent of the Hub. Step 5a performs the downloads.

## 3. Data

LeRobot datasets are read from `$HF_LEROBOT_HOME/<repo id>` (default
`~/.cache/huggingface/lerobot`). Export the same `HF_LEROBOT_HOME` in every later command.

- **LIBERO**: `physical-intelligence/libero` and `IPEC-COMMUNITY/libero_90_no_noops_lerobot`
  download automatically on first use.
- **ABC-130k** (bimanual YAM, `XDOF/ABC-130k`): convert episodes into the LeRobot layout with
  `scripts/data/convert_abc130k_to_lerobot.py`. [docs/data.md](docs/data.md) has the commands,
  including validation splits and the episode selection used by the configs. A small first
  conversion:

  ```bash
  python scripts/data/convert_abc130k_to_lerobot.py --src hf:fold_and_stack_the_t_shirts --max-episodes 50
  ```

**Check:**

```bash
python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata as M; \
m = M('abc130k/fold_and_stack_the_t_shirts'); print(m.total_episodes, m.total_frames, m.fps)"
```

## 4. Checkpoints

[docs/checkpoints.md](docs/checkpoints.md) lists the released checkpoints (the Hugging Face links
are added when they are published):

```bash
trvam download <hf repo> --subfolder <dir> --out checkpoints/   # or in a config: checkpoint: hf://<repo>/<dir>
```

**Check:** the step directory contains `model.safetensors`, `metadata.pt`,
`assets/**/norm_stats.json`, and `video_model_full.pt` or `lora_adapters/`.

## 5. GPU smoke tests: evaluation, training, serving

The commands use a YAM checkpoint with ABC-130k data. `CKPT` is a checkpoint step directory and
`OUT` a scratch directory.

**a) Open-loop evaluation.** The first run also downloads the pretrained models.

```bash
trvam infer configs/infer/abc130k/yam_pp24.yaml --set checkpoint=$CKPT \
    --dataset abc130k/fold_and_stack_the_t_shirts --episodes 0 --max-steps 3 --out $OUT/infer
```

**Check:** exit code 0; the printed summary and `$OUT/infer/report.json` show `all_finite: true`
and a finite `action_mse`.

**b) Training.** Ten steps of fine-tuning from the checkpoint on T-shirt folding:

```bash
TR_FINETUNE_FROM=$CKPT trvam train configs/train/abc130k/finetune_from_checkpoint.yaml --gpus 1 \
    --set data.action_dataset=videoVLA_abc130k_folding_short --set data.video_dataset=abc130k_folding \
    --set stages.0.steps=10 --set stages.0.save_interval=5 \
    --set output_dir=$OUT/train --set logging.wandb=false
```

The default batch sizes (video 2, action 8) need about 135 GB of GPU memory. On smaller GPUs add
`--set optim.video_batch_size=1 --set optim.action_batch_size=2 --set optim.gradient_checkpointing=true`.
The first step takes several minutes (data loaders, text encoder).

**Check:** exit code 0 and checkpoints `5/` and `10/` in
`$OUT/train/videoVLA_abc130k_folding_short/abc130k_finetune/`. Repeat the command with
`--set stages.0.steps=15`; the log shows `Resumed from step 10` and
`Learning-rate cycle from step 10`, and checkpoint `15/` appears.

**c) Evaluate the trained checkpoint.** Repeat 5a with
`--set checkpoint=$OUT/train/videoVLA_abc130k_folding_short/abc130k_finetune/15`.

**d) Serving.**

```bash
setsid nohup trvam serve configs/infer/abc130k/yam_pp24.yaml --set checkpoint=$CKPT --port 8000 \
    > $OUT/serve.log 2>&1 < /dev/null &
SERVER=$!
python scripts/eval/check_server.py configs/infer/abc130k/yam_pp24.yaml --port 8000 \
    --dataset abc130k/fold_and_stack_the_t_shirts --episode 0 --frames 0 50
kill $SERVER
```

**Check:** `check_server.py` exits 0 and prints `ok` with an action shape of `(50, 14)` for each
frame. It waits until the server has loaded the model.

## 6. Common tasks

- **Inspect a recipe:** `trvam show <config>` prints the resolved config and the exact trainer
  commands. Override fields with `--set key=value` (lists as `[a,b]`); `${VAR:-default}` in the
  YAML reads environment variables.
- **Fine-tune on ABC-130k tasks:** as in 5b, with the task group of
  `src/temporal_ratio/data/abc130k.py` (`TASK_GROUPS`, e.g. `folding`, `pen_caps`, `assembly`,
  `fold_pen_bottle`). One epoch over the 50-step action chunks is frames / 50 / action batch size
  steps; [docs/pipeline_abc130k.md](docs/pipeline_abc130k.md) walks through a complete example.
- **Train longer:** repeat a training command with a larger `stages.0.steps`. The run resumes from
  its latest checkpoint with a new learning-rate cycle. Add `stages.0.keep_every=<steps per
  epoch>` to keep every epoch's checkpoint ([docs/training.md](docs/training.md#continuing-a-run)).
- **Train from the pretrained base model:** the presets in `configs/train/` (table in the README)
  default to 2 nodes × 8 GPUs; `--gpus N` runs them on one node.
- **Evaluate:** `trvam infer configs/infer/abc130k/fold_pen_bottle.yaml --set checkpoint=<ckpt>`
  covers the first 200 frames of 5 validation episodes per task; `fold_pen_bottle_heldout.yaml`
  covers whole validation episodes. `report.json` gives `action_mse` / `action_l1` in action units,
  and `action_rmse_std` / `first_k_rmse_std` in units of the dataset's per-dimension action standard
  deviation, over the whole chunk and over its first `offline.first_k` actions.
- **Serve to a robot:** `trvam serve <inference config> --set checkpoint=<ckpt> --port 8000`.
  Clients use `packages/openpi-client`; observation keys are in
  [docs/inference.md](docs/inference.md).
- **Closed-loop LIBERO:** set up the simulator ([docs/install.md](docs/install.md)), then
  `trvam eval-libero configs/eval/libero/id.yaml --set inference.checkpoint=<ckpt>`.

## 7. Troubleshooting

| symptom | fix |
|---|---|
| HTTP 401 or 403 while downloading `nvidia/...` | the user accepts the model license on the Hub and runs `hf auth login` |
| missing files with `HF_HUB_OFFLINE=1` | unset it for one run so the files download |
| CUDA out of memory | check `nvidia-smi` for other jobs; lower `optim.video_batch_size` / `optim.action_batch_size`, or set `optim.gradient_checkpointing=true` |
| `... already has checkpoints` | the run directory has checkpoints: keep `resume: true` (default) to continue, or set `overwrite=true` to start over |
| normalization statistics not found | fine-tuning reads them from the checkpoint (`data.norm_stats`); for new data run `scripts/data/compute_norm_stats.py --config-name <policy config> --fast` |
| `Address already in use` from `trvam serve` | choose another `--port` |
| slow video decoding | optionally install `torchcodec` ([docs/install.md](docs/install.md#video-decoding)) |
