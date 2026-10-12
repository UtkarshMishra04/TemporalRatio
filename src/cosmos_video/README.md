# `cosmos_video`: vendored Cosmos-Predict2.5

A subset of [NVIDIA Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5)
(package `cosmos_predict2`, Apache-2.0), renamed to `cosmos_video` so it can be installed next to
an upstream checkout. `temporal_ratio.models.video_model.VideoProcessor` loads the 2B Video2World
model (DiT, Wan2.1 VAE, Cosmos-Reason1 text encoder) through it.

Changes with respect to upstream:

- `_src/predict2/inference/video2world.py` (`Video2WorldInference.eval_vid2world`) and
  `_src/predict2/models/text2world_model_rectified_flow.py` (`generate_eval_velocity_from_batch`):
  return intermediate DiT block outputs (`intermediate_feature_ids`) and their timesteps, stop the
  denoising loop early (`target_denoising_steps`) and optionally skip classifier-free guidance
  (`skip_cfg`); these provide the features read by the action head.
- `experiments/base/cosmos_lerobot_assets_flat.py`: the two experiments used here (LIBERO and
  bimanual YAM with horizontally tiled cameras).
- `_src/imaginaire/utils/checkpoint_db.py`: tokenizer revision updated to a revision that exists on
  the Hugging Face Hub.
- `_src/reason1/configs/default/model_config_qwen.py`: falls back to SDPA attention when
  flash-attention 2 is not installed.
- `_src/predict2/utils/model_loader_simple.py`: loads checkpoints with `map_location="cpu"`.
- Import paths (`cosmos_predict2` -> `cosmos_video`) and removal of modules not needed here.
