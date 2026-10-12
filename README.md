# TemporalRatio

[![CI](https://github.com/UtkarshMishra04/TemporalRatio/actions/workflows/ci.yml/badge.svg)](https://github.com/UtkarshMishra04/TemporalRatio/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2607.08127-b31b1b.svg)](https://arxiv.org/abs/2607.08127)
[![Project page](https://img.shields.io/badge/project-page-blue.svg)](https://umishra.me/temporal-ratio/)
[![Checkpoints](https://img.shields.io/badge/%F0%9F%A4%97%20checkpoints-coming%20soon-yellow.svg)](docs/checkpoints.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

Code for **Understanding and Mitigating the Video-Action Generalization Gap via Temporal Ratio**
(Utkarsh A. Mishra, Yongxin Chen, Danfei Xu, Yang Liu, Xi Chen, Jiayuan Mao).
[Paper](https://arxiv.org/abs/2607.08127) · [Project page](https://umishra.me/temporal-ratio/)

Video-action models (VAMs) predict robot actions from the internal features of a video generation
model. This repository trains and runs such a model: a
[Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) 2B Video2World DiT whose
block-19 features condition a π0-style Gemma-300M flow-matching action head. It also implements
the paper's analysis and method:

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

**Setting up with a coding agent?** Point it to [AGENT.md](AGENT.md): installation, models, data,
checkpoints and GPU smoke tests of training, evaluation and serving, with a check after each step.

## Repository layout

```
configs/                    experiment configs (YAML) — see docs/configs.md
  base/                     model family defaults (Cosmos-Predict2.5 2B + Gemma-300M, LoRA r32)
  train/{libero,abc130k}/   training recipes (joint / two-stage / three-stage, LoRA or full, ...)
  infer/{libero,abc130k}/   inference settings (video steps, guidance, TR logging)
  eval/libero/              closed-loop LIBERO evaluation
src/temporal_ratio/
  cli.py                    the `trvam` command
  config/                   config schema, loader, translation to trainer arguments, dataset registry
  models/                   video model wrapper, action head, joint video-action model
  data/                     LeRobot action data, video datasets, transforms, ABC-130k task lists
  training/                 trainers (standard and dual-resolution) and their arguments
  policies/                 inference policy (video model + action head), robot input/output transforms
  guidance/                 Temporal Ratio and TR-adaptive guidance
  inference.py, offline.py  policy loading from configs; open-loop evaluation
src/cosmos_video/           vendored Cosmos-Predict2.5 (see its README for local changes)
scripts/                    data conversion, norm stats, LIBERO evaluation, TR analysis, setup
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
[docs/install.md](docs/install.md) for details, LIBERO simulation setup and CPU-only installs.

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

**Open-loop inference** on recorded episodes. This compares predicted with recorded action
chunks, logs TR per action-expert layer and optionally renders the predicted future:

```bash
trvam infer configs/infer/abc130k/yam_pp24.yaml \
    --set checkpoint=<checkpoint step dir> \
    --dataset abc130k/fold_and_stack_the_t_shirts --episodes 0 --max-steps 5 \
    --out results/infer/yam
# ... and with video generation (full video denoising + VAE decode, written as mp4):
trvam infer configs/infer/abc130k/yam_pp24.yaml --set checkpoint=<ckpt> --set generate_video=true ...
```

**Serve** a policy to a robot over websockets (the
[openpi-client](packages/openpi-client) protocol):

```bash
trvam serve configs/infer/libero/no_guidance.yaml --set checkpoint=<ckpt> --port 8000
```

**Closed-loop LIBERO** evaluation, with or without TR-adaptive guidance:

```bash
trvam eval-libero configs/eval/libero/id.yaml  --set inference.checkpoint=<ckpt>   # 4 ID suites
trvam eval-libero configs/eval/libero/ood.yaml --set inference.checkpoint=<ckpt>   # 3 OOD suites
```

Evaluation reports also give the action errors in units of the evaluated dataset's per-dimension
action standard deviation, over the whole chunk and over its first `offline.first_k` actions;
`configs/infer/abc130k/fold_pen_bottle_heldout.yaml` evaluates whole validation episodes.
`scripts/eval/check_server.py` sends recorded frames to a running server and checks its answers.
See [docs/inference.md](docs/inference.md) for the inference pipeline, observation formats and
timings.

## Training

Every run is an experiment config: model, DiT adaptation (`lora` with rank/targets, `full` or
`frozen`), data, optimizer and one or more **stages** (`joint_full`, `video_only`,
`frozen_video`, `joint_detach`). `trvam show` prints the resolved config and the exact trainer
commands; `trvam train` runs them, chaining stages through their latest checkpoints.

```bash
trvam show  configs/train/libero/two_stage_lora_r32.yaml
trvam train configs/train/libero/joint_lora_r32.yaml                       # 2 nodes x 8 GPUs by default
trvam train configs/train/libero/joint_lora_r32.yaml --set distributed.nnodes=1 \
    --set adaptation.lora.target_modules=[q_proj,k_proj,v_proj] --set stages.0.steps=50000
trvam train configs/train/libero/joint_full.yaml                           # full fine-tuning of the DiT
TR_FINETUNE_FROM=<ckpt> trvam train configs/train/libero/finetune_from_checkpoint.yaml
```

| preset (`configs/train/...`) | recipe | released checkpoint |
|---|---|---|
| `libero/joint_lora_r{16,32,64}` | joint video + action training, LoRA on the DiT | `joint_r*` |
| `libero/joint_full` | joint training, full DiT fine-tuning | `full_joint` |
| `libero/joint_93f_lora_r{16,32,64}`, `libero/joint_93f_full` | joint training with 93-frame video clips | `joint_93f_r*`, `full_joint_93f` |
| `libero/two_stage_lora_r{16,32,64}`, `libero/two_stage_full` | video-only adaptation, then the action head on the frozen DiT | `2stage_r*`, `full_2stage` |
| `libero/two_stage_93f_lora_r32`, `libero/two_stage_93f_full` | as above with 93-frame video clips | `2stage_93f_r*`, `full_2stage_93f` |
| `libero/two_stage_lora_r32_16k`, `libero/three_stage_93f_lora_r32` | schedules of the Temporal Ratio experiments | |
| `abc130k/pp24_joint_lora_r32` | joint training on the 22 ABC-130k tasks (dual-resolution trainer) | YAM `878500` |
| `abc130k/fold_pen_bottle_joint_lora_r32` | the same recipe on T-shirt folding, pen caps and bottle caps | |
| `abc130k/folding_action_only` | action head only, frozen DiT (folding skill) | |
| `abc130k/skill_finetune_93f` | per-skill joint fine-tuning with 93-frame video clips | |
| `*/finetune_from_checkpoint` | continue from a released checkpoint | |

The presets reproduce the trainer arguments of the original runs exactly; the test suite checks
all of them against the original launch commands. See [docs/configs.md](docs/configs.md) and
[docs/training.md](docs/training.md).

Every checkpoint holds the full training state, and runs resume from their latest checkpoint by
default. To train longer, repeat a command with a larger `stages.0.steps`: the run continues with
a new learning-rate cycle over the added steps. `stages.0.keep_every=<steps>` keeps, for example,
every epoch's checkpoint ([docs/training.md](docs/training.md#continuing-a-run)).

## Data

- **LIBERO**: actions from [`physical-intelligence/libero`](https://huggingface.co/datasets/physical-intelligence/libero)
  and video from it plus [`IPEC-COMMUNITY/libero_90_no_noops_lerobot`](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_90_no_noops_lerobot).
  Both are downloaded on first use; normalization statistics ship in `assets/`.
- **ABC-130k** ([`XDOF/ABC-130k`](https://huggingface.co/datasets/XDOF/ABC-130k), bimanual YAM):
  `scripts/data/convert_abc130k_to_lerobot.py` converts the MCAP episodes into the LeRobot layout
  the YAM checkpoints were trained on. The `fold_pen_bottle` configs use T-shirt folding, pen-cap
  opening and bottle-cap unscrewing; compute normalization statistics after converting with
  `scripts/data/compute_norm_stats.py --config-name videoVLA_abc130k_fold_pen_bottle_short --fast`.

See [docs/data.md](docs/data.md), including how to add a robot, and
[docs/pipeline_abc130k.md](docs/pipeline_abc130k.md) for an end-to-end example (data, evaluation,
fine-tuning, serving) on ABC-130k T-shirt folding, pen caps and bottle caps.

## Temporal Ratio and guidance

TR at action-expert layer *l* is the attention mass from action queries on future latent frames
1…T−1 divided by the mass on the current frame 0, averaged over heads and measured at the first
flow step. Guided inference combines velocities
`v = v_c + w_lang (v_c − v_∅) + w_plan (v_plan − v_c)` with weights scaled by
`s = clip(TR_k / TR_0 − 1, 0, s_max)`. The configs `configs/infer/libero/{flat,tr_adaptive}.yaml`
select flat and TR-adaptive guidance; [docs/temporal_ratio.md](docs/temporal_ratio.md) gives the
details and the analysis scripts.

## Tests

CI ([workflow](.github/workflows/ci.yml)) runs on every push and pull request:

- **unit tests** on CPU with Python 3.11 and 3.12. They cover config validation, the trainer arguments
  of every preset against the original launch commands, dataset task lists, evaluation metrics,
  run continuation and checkpoint retention, TR and guidance math, and a scan for credentials and
  internal references;
- **CLI dry runs** of every training preset and of the LIBERO evaluation;
- **lint** (ruff) and a **secret scan** of the full git history (gitleaks).

```bash
uv venv --python 3.11 && uv pip install -e ".[test]" && .venv/bin/pytest -q   # no GPU needed
```

CI has no GPUs; the GPU paths were verified on an NVIDIA GB300:

| stage | verified |
|---|---|
| installation | `scripts/setup/install.sh` in a fresh environment (aarch64, CUDA 13) |
| training | joint, two-stage and action-only recipes; fine-tuning released LIBERO and YAM checkpoints; single- and multi-task ABC-130k fine-tuning; continuing a run for more epochs |
| open-loop evaluation | LIBERO and YAM checkpoints with and without video generation, ABC-130k validation episodes |
| serving | `trvam serve` with websocket clients: the closed-loop LIBERO evaluator and `scripts/eval/check_server.py` |
| closed-loop LIBERO | `libero_spatial` with and without TR-adaptive guidance |

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
