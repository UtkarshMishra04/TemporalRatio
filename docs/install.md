# Installation

## GPU environment (training and inference)

Requirements: Linux (x86_64 or aarch64), an NVIDIA driver for CUDA 13 (or CUDA 12.8 with
`TORCH_INDEX=https://download.pytorch.org/whl/cu128`), [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/UtkarshMishra04/TemporalRatio.git
cd TemporalRatio
bash scripts/setup/install.sh            # -> .venv
source .venv/bin/activate
```

`scripts/setup/install.sh` creates a Python 3.11 venv and installs:

1. `torch==2.11.0` and `torchvision==0.26.0` from `$TORCH_INDEX` (CUDA 13 by default, required
   for Blackwell / GB300 GPUs);
2. the pinned stack in `requirements/gpu.txt` (transformers 4.53.2, LeRobot at a fixed revision
   with `datasets<3`, the vendored Cosmos runtime, CPU-only JAX used by the config utilities);
3. `packages/openpi-client` (without its `numpy<2` pin) and this repository in editable mode;
4. the modified Gemma / PaliGemma / SigLIP modules of the action expert
   (`scripts/setup/patch_transformers.py` copies them over the installed `transformers`).

flash-attention, Transformer Engine and NATTEN are not required; the video model uses PyTorch
SDPA. The setup is verified on NVIDIA GB300 (aarch64, driver for CUDA 13); on aarch64 `decord2`
and `megatron-core` 0.16 provide the wheels missing for the versions used originally.

### Models

On first use the following are downloaded into the Hugging Face cache (accept the model licenses
on the Hub and run `hf auth login` if your account needs it):

| repository | files | size |
|---|---|---|
| `nvidia/Cosmos-Predict2.5-2B` | `base/pre-trained/*_ema_bf16.pt` (DiT), `tokenizer.pth` (Wan2.1 VAE) | 4.6 GB |
| `nvidia/Cosmos-Reason1-7B` | text encoder | 16.6 GB |

Afterwards `HF_HUB_OFFLINE=1` runs fully offline. `$COSMOS_BASE_CKPT` points the trainers and
the policy at a local copy of the base DiT.

### Video decoding

LeRobot decodes dataset videos with `torchcodec` if installed, otherwise with PyAV. LeRobot's
PyAV path relies on `torchvision.io.VideoReader`, which torchvision 0.26 removed; the release
replaces it with an equivalent PyAV decoder (`temporal_ratio.data.video_decode`), so no extra
packages are needed. For faster decoding install `torchcodec==0.11.0` (on aarch64 together with
`nvidia-npp-cu13`, adding `site-packages/nvidia/cu13/lib` to `LD_LIBRARY_PATH`).

## LIBERO simulation (closed-loop evaluation only)

The evaluator uses a LIBERO fork that adds the out-of-distribution suites
(`libero_{spatial,object,goal}_ood`):

```bash
git clone -b libero_ood https://github.com/UtkarshMishra04/LIBERO.git third_party/libero
uv pip install --no-deps -e third_party/libero
uv pip install robosuite==1.4.0 mujoco==3.4.0 bddl==1.0.1 easydict gym==0.25.2 future
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero   # `libero` is a namespace package
export MUJOCO_GL=egl            # or osmesa without a GPU display stack
```

The in-distribution suites are also available from upstream
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO); this setup is verified on GB300
(aarch64) with EGL rendering.

## CPU only (configs, tests, dry runs)

```bash
uv venv --python 3.11 && uv pip install -e ".[test]"
.venv/bin/pytest -q
.venv/bin/trvam show configs/train/libero/two_stage_lora_r32.yaml
```
