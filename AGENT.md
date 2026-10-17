# AGENT.md

Instructions for coding agents (Claude Code, Codex, Cursor, ...) that set up and run the inference
server of this repository for a user (training and evaluation are on the `master` branch). Work through the steps in order. Each step ends with a **Check**; do not go
on until it passes. Human-oriented documentation: [README.md](README.md) and [docs/](docs/).

## Rules

- Never print, log, commit or write credentials (Hugging Face, GitHub tokens) into this
  repository. Hugging Face access goes through `hf auth login`, which the user runs, or an
  `HF_TOKEN` the user has exported. The test suite and CI scan for secrets.
- The GPU may be shared. Run `nvidia-smi` before starting a GPU job, and never stop processes you
  did not start. Inference needs about 30 GB of GPU memory.
- Start long jobs detached and poll their logs:
  `setsid nohup <command> > <log> 2>&1 < /dev/null &`.
- After changing code or configs, run `pytest -q`.
- Ask the user before large downloads (tens of GB), before deleting checkpoints, and before
  pushing.

## 1. Environment

Requirements: Linux (x86_64 or aarch64), an NVIDIA GPU with a driver for CUDA 13 (for CUDA 12.8 set
`TORCH_INDEX=https://download.pytorch.org/whl/cu128`), [uv](https://docs.astral.sh/uv/), and about
60 GB of disk for the environment, models and checkpoints.

```bash
git clone https://github.com/UtkarshMishra04/TemporalRatio.git && cd TemporalRatio
bash scripts/setup/install.sh            # creates .venv (Python 3.11); see docs/install.md
source .venv/bin/activate
```

**Check:**

```bash
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())"
trvam --help        # subcommands: show, serve, download
pytest -q           # all tests pass
```

Without a GPU (configs and tests only): `uv venv --python 3.11 && uv pip install -e ".[test]"`.

## 2. Pretrained models

The first server start downloads `nvidia/Cosmos-Predict2.5-2B` (DiT and VAE, 4.6 GB) and
`nvidia/Cosmos-Reason1-7B` (text encoder, 16.6 GB) into the Hugging Face cache. The user must
accept both licenses on the Hub. If a download fails with HTTP 401 or 403, ask the user to accept
the licenses and run `hf auth login`. Once downloaded, `export HF_HUB_OFFLINE=1` makes runs
independent of the Hub. Step 4 performs the downloads.

## 3. Checkpoints

[docs/checkpoints.md](docs/checkpoints.md) lists the released checkpoints (the Hugging Face links
are added when they are published):

```bash
trvam download <hf repo> --subfolder <dir> --out checkpoints/   # or in a config: checkpoint: hf://<repo>/<dir>
```

**Check:** the step directory contains `model.safetensors`, `metadata.pt`,
`assets/**/norm_stats.json`, and `video_model_full.pt` or `lora_adapters/`.

## 4. Serve

`CKPT` is a checkpoint step directory and `OUT` a scratch directory. The commands use a YAM
checkpoint; for LIBERO use `configs/infer/libero/no_guidance.yaml` (or `tr_adaptive.yaml`).

```bash
setsid nohup trvam serve configs/infer/abc130k/yam_pp24.yaml --set checkpoint=$CKPT --port 8000 \
    --save-video $OUT/server_videos > $OUT/serve.log 2>&1 < /dev/null &
```

**Check:** `$OUT/serve.log` ends with `serving <checkpoint> ... on 0.0.0.0:8000` after the model has
loaded (60-80 s once the pretrained models are cached).

## 5. Check the server

```bash
python scripts/check_server.py --port 8000 --random --queries 2
python scripts/check_server.py --port 8000 --random --video-dir $OUT/client_videos
```

**Check:** both commands exit 0 and print `ok` with an action shape of `(50, 14)` (`(17, 7)` for
LIBERO) for each query; the second writes `query_00000.mp4` and `query_00000.png` to
`$OUT/client_videos` and the server writes `query_<n>.mp4` to `$OUT/server_videos`. The client
waits until the server has loaded the model. Stop the server afterwards (`pkill -f "trvam serve"`
only if you started it).

## 6. Common tasks

- **Inspect a config:** `trvam show <config>` prints the resolved config. Override fields with
  `--set key=value` (lists as `[a,b]`); `${VAR:-default}` in the YAML reads environment variables.
- **Real observations:** `scripts/check_server.py --images <one image per camera> --state ... --prompt ...`;
  observation keys and camera order are in [docs/inference.md](docs/inference.md).
- **TR-adaptive guidance:** serve `configs/infer/libero/tr_adaptive.yaml`
  ([docs/temporal_ratio.md](docs/temporal_ratio.md)).
- **Python API:** `temporal_ratio.inference.load_policy` ([docs/inference.md](docs/inference.md)).

## 7. Troubleshooting

| symptom | fix |
|---|---|
| HTTP 401 or 403 while downloading `nvidia/...` | the user accepts the model license on the Hub and runs `hf auth login` |
| missing files with `HF_HUB_OFFLINE=1` | unset it for one run so the files download |
| CUDA out of memory | check `nvidia-smi` for other jobs; the server needs about 30 GB |
| normalization statistics not found | point `norm_stats` (or `$NORM_STATS_PATH`) at the checkpoint's `assets/` |
| `Address already in use` from `trvam serve` | choose another `--port` |
