# Differences from the research code

The training recipes are unchanged: every preset produces exactly the trainer arguments of the
original launch commands (`tests/test_train_translation.py`). The following behaviours differ.

**Guidance**
- TR-adaptive guidance switches guidance off when the adaptive scale is ~0. The research code kept
  the flat base weights in that case; `guidance.legacy_flat_floor: true` restores it.

**Inference**
- The LIBERO evaluator uses the checkpoint's own DiT (`video_model_full.pt` or `lora_adapters/`)
  unless `--vid2world_ckpt` is given; previously it fell back to a fixed local video-model path
  when the flag was omitted.
- The policy is built from the requested named config (`policy_config`) instead of the config
  name stored in `metadata.pt`, and normalization statistics fall back to the single
  `norm_stats.json` inside the checkpoint. Released checkpoints trained under internal dataset
  names therefore load with the public configs.
- The video setup (resolution, camera order, Cosmos experiment) is declared by each named config
  (`video_setup`) instead of being inferred from substrings of the config name.
- The LIBERO evaluator also runs with upstream LIBERO (in-distribution suites), exits non-zero when
  a rollout crashes, and can be restricted to suites / tasks (`eval.suites`, `eval.task_ids`).
- The policy server runs inference off the websocket event loop, so long (cold) queries no longer
  drop the connection.

**Training**
- When the base checkpoint is `base`, the pretrained Cosmos-Predict2.5 2B DiT is resolved through
  the pinned Cosmos checkpoint registry; previously a missing local cache fell back to downloading
  a different model (Cosmos-Predict2 2B).
- W&B runs use the configured project and `$WANDB_ENTITY`; checkpoint uploads are optional
  (`$CHECKPOINT_SYNC_URI`).
- `--video_frame_skip` (dual-resolution trainer) is checked against the video dataset's frame skip
  instead of being ignored.

**Data**
- The LIBERO video mixture defaults to the public datasets; the 10% failed-rollout videos of the
  original runs are optional (`libero_with_failures`).
- ABC-130k replaces the internal bimanual datasets (22 of the 24 tasks; `data/abc130k.py`).
- On torchvision >= 0.26, LeRobot video decoding uses a PyAV implementation
  (`temporal_ratio.data.video_decode`).

**Environment**
- The vendored Cosmos code downloads the tokenizer from a revision that exists on the Hub, uses
  SDPA attention without flash-attention, loads checkpoints to CPU first, and no longer downloads
  the 14B model at import time.
