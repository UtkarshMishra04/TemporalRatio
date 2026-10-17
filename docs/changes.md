# Differences from the research code

The following inference behaviours differ from the research code (training and evaluation
differences are listed on the `master` branch).

**Guidance**
- TR-adaptive guidance switches guidance off when the adaptive scale is ~0. The research code kept
  the flat base weights in that case; `guidance.legacy_flat_floor: true` restores it.

**Inference**
- The policy uses the checkpoint's own DiT (`video_model_full.pt` or `lora_adapters/`);
  previously the LIBERO evaluator fell back to a fixed local video-model path.
- The policy is built from the requested named config (`policy_config`) instead of the config
  name stored in `metadata.pt`, and normalization statistics fall back to the single
  `norm_stats.json` inside the checkpoint. Released checkpoints trained under internal dataset
  names therefore load with the public configs.
- The video setup (resolution, camera order, Cosmos experiment) is declared by each named config
  (`video_setup`) instead of being inferred from substrings of the config name.
- The policy server runs inference off the websocket event loop, so long (cold) queries no longer
  drop the connection.

**Environment**
- The vendored Cosmos code downloads the tokenizer from a revision that exists on the Hub, uses
  SDPA attention without flash-attention, loads checkpoints to CPU first, and no longer downloads
  the 14B model at import time.
