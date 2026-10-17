# TemporalRatio

[![CI](https://github.com/UtkarshMishra04/TemporalRatio/actions/workflows/ci.yml/badge.svg)](https://github.com/UtkarshMishra04/TemporalRatio/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2607.08127-b31b1b.svg)](https://arxiv.org/abs/2607.08127)
[![Project page](https://img.shields.io/badge/project-page-blue.svg)](https://umishra.me/temporal-ratio/)
[![Checkpoints](https://img.shields.io/badge/%F0%9F%A4%97%20checkpoints-coming%20soon-yellow.svg)](docs/checkpoints.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

Code for **Understanding and Mitigating the Video-Action Generalization Gap via Temporal Ratio**
(Utkarsh A. Mishra, Yongxin Chen, Danfei Xu, Yang Liu, Xi Chen, Jiayuan Mao).
[Paper](https://arxiv.org/abs/2607.08127) · [Project page](https://umishra.me/temporal-ratio/)

> **This `eval` branch contains only the inference server**: loading released checkpoints, serving
> them over websockets (with optional predicted-video output) and the Python API. Training,
> open-loop and closed-loop evaluation, data conversion and the TR analysis scripts are on
> [`master`](https://github.com/UtkarshMishra04/TemporalRatio/tree/master).

Video-action models (VAMs) predict robot actions from the internal features of a video generation
model. This repository runs such a model: a
[Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) 2B Video2World DiT whose
block-19 features condition a π0-style Gemma-300M flow-matching action head. It also implements
the paper's inference-time method:

- **Temporal Ratio (TR)**: how much the action head's attention goes to *future* predicted latent
  frames relative to the current frame.
- **TR-adaptive guidance**: inference-time classifier-free guidance on the action head whose
  strength follows TR, amplifying language and long-horizon video conditioning when the policy
  relies on its video rollout.

```
 camera frames ─┐                                     ┌─> predicted future video (optional)
 (tiled)        ├─> Cosmos-Predict2.5 2B DiT ─────────┤
 task prompt ───┘   (Reason1 text encoder, Wan VAE)   └─> block-19 features ─┐
                                                                             ├─> Gemma-300M action head ─> action chunk
 robot state ────────────────────────────────────────────────────────────────┘      (flow matching)
```

**Setting up with a coding agent?** Point it to [AGENT.md](AGENT.md): installation, models,
checkpoints, serving and a client check, with a check after each step.

## Repository layout

```
configs/infer/{libero,abc130k}/   inference settings (video steps, guidance, TR logging) — see docs/configs.md
src/temporal_ratio/
  cli.py                    the `trvam` command
  config/                   config schema, loader, dataset registry
  models/                   video model wrapper, action head
  data/                     transforms, normalization, ABC-130k task lists
  training/                 optimizer, sharding and weight-loading helpers used by the model configs
  policies/                 inference policy (video model + action head), robot input/output transforms
  guidance/                 Temporal Ratio and TR-adaptive guidance
  serving/                  websocket policy server
  inference.py              policy loading from configs, server-side wrappers
src/cosmos_video/           vendored Cosmos-Predict2.5 (see its README for local changes)
packages/openpi-client/     websocket client
scripts/check_server.py     client check of a running server
scripts/setup/              installation
tests/                      CPU test suite (runs in CI)
```

## Installation

Python 3.11 and an NVIDIA GPU; inference needs about 30 GB of GPU memory (of which 16 GB is the
text encoder). The environment is verified on NVIDIA GB300 (aarch64, CUDA 13).

```bash
git clone https://github.com/UtkarshMishra04/TemporalRatio.git
cd TemporalRatio
bash scripts/setup/install.sh          # creates .venv; TORCH_INDEX selects the CUDA build
source .venv/bin/activate
```

The first run downloads the pretrained models from Hugging Face: `nvidia/Cosmos-Predict2.5-2B`
(DiT and tokenizer, 4.6 GB) and `nvidia/Cosmos-Reason1-7B` (text encoder, 16.6 GB). See
[docs/install.md](docs/install.md) for details and CPU-only installs.

## Checkpoints

| checkpoints | robot / data | Hugging Face |
|---|---|---|
| LIBERO policies: joint, two-stage and full DiT fine-tuning, 17- and 93-frame video, LoRA ranks 16 to 64 | Franka (simulation), LIBERO | 🤗 *link coming soon* |
| Bimanual YAM policies trained on 24 tasks (22 of them in ABC-130k) | YAM, ABC-130k | 🤗 *link coming soon* |

<!-- Fill in the Hugging Face repository links here and in docs/checkpoints.md when they are published. -->

A checkpoint is a step directory with `model.safetensors` (action head), `video_model_full.pt`
(fine-tuned DiT) and/or `lora_adapters/`, `metadata.pt` and `assets/**/norm_stats.json`.
[docs/checkpoints.md](docs/checkpoints.md) lists every released checkpoint with its training
preset. `trvam download <repo> --subfolder <dir>` fetches one, and configs also accept
`checkpoint: hf://<owner>/<repo>/<dir>`.

## Quickstart

**Serve** a policy to a robot over websockets (the
[openpi-client](packages/openpi-client) protocol):

```bash
trvam serve configs/infer/libero/no_guidance.yaml --set checkpoint=<ckpt> --port 8000
trvam serve configs/infer/abc130k/yam_pp24.yaml --set checkpoint=<ckpt> --save-video results/videos
```

A request can ask for the predicted future video with `"generate_video": True` in the observation;
the response then carries `future_video` (T x H x W x 3 uint8, cameras tiled left to right), and
`--save-video DIR` also writes it to `DIR/query_<n>.mp4`. Video generation roughly doubles the
query latency (about 1.5 s action-only and 3 s with video on a GB300).

**Check** a running server with images (one per camera) or random inputs:

```bash
python scripts/check_server.py --port 8000 --random --queries 3
python scripts/check_server.py --port 8000 --images top.png left_wrist.png right_wrist.png \
    --prompt "fold the t-shirt" --video-dir results/check_server
```

**Python API**: `load_policy(loader.load_inference(cfg, ["checkpoint=..."]))` returns the policy;
`policy.infer(obs)` gives the action chunk. See [docs/inference.md](docs/inference.md) for the
inference pipeline, observation formats and timings.

## Temporal Ratio and guidance

TR at action-expert layer *l* is the attention mass from action queries on future latent frames
1…T−1 divided by the mass on the current frame 0, averaged over heads and measured at the first
flow step. Guided inference combines velocities
`v = v_c + w_lang (v_c − v_∅) + w_plan (v_plan − v_c)` with weights scaled by
`s = clip(TR_k / TR_0 − 1, 0, s_max)`. The configs `configs/infer/libero/{flat,tr_adaptive}.yaml`
select flat and TR-adaptive guidance; [docs/temporal_ratio.md](docs/temporal_ratio.md) gives the
details.

## Tests

CI ([workflow](.github/workflows/ci.yml)) runs on every push and pull request:

- **unit tests** on CPU with Python 3.11 and 3.12. They cover config validation, dataset task
  lists, TR and guidance math, the server-side video options, and a scan for credentials and
  internal references;
- **CLI**: `trvam show` of every inference config;
- **lint** (ruff) and a **secret scan** of the full git history (gitleaks).

```bash
uv venv --python 3.11 && uv pip install -e ".[test]" && .venv/bin/pytest -q   # no GPU needed
```

CI has no GPUs; the GPU paths were verified on an NVIDIA GB300 (on `master`):

| stage | verified |
|---|---|
| installation | `scripts/setup/install.sh` in a fresh environment (aarch64, CUDA 13) |
| inference | LIBERO and YAM checkpoints with and without video generation |
| serving | `trvam serve` with websocket clients |

[docs/checkpoints.md](docs/checkpoints.md#verified) names the checkpoints used.
[docs/changes.md](docs/changes.md) lists where the release deliberately differs from the research
code.

## Citation

```bibtex
@inproceedings{mishra2026temporalratio,
  title     = {Understanding and Mitigating the Video-Action Generalization Gap via Temporal Ratio},
  author    = {Mishra, Utkarsh A. and Chen, Yongxin and Xu, Danfei and Liu, Yang and Chen, Xi and Mao, Jiayuan},
  booktitle = {10th Annual Conference on Robot Learning},
  year      = {2026},
  url       = {https://openreview.net/forum?id=mlF6DhhBNF}
}
```

## License

Apache 2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). The repository builds on
[openpi](https://github.com/Physical-Intelligence/openpi) and
[Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5). Model weights are subject
to their own licenses (NVIDIA Open Model License, Gemma Terms of Use).
