# Inference configs

The policy is configured by YAML inference configs under `configs/infer/`. A config is validated
against the dataclasses in `src/temporal_ratio/config/schema.py` (unknown keys, wrong types and
inconsistent settings are errors).

```bash
trvam show  <infer.yaml> [--set key=value ...]   # resolved config
trvam serve <infer.yaml> [--set ...] [--port N]  # websocket policy server
```

## Composition

- `_base_: <path>` (relative to the file) inherits another config; mappings are merged
  recursively and lists are replaced. Nested mappings may have their own `_base_`.
- `${VAR}` and `${VAR:-default}` are replaced by environment variables; a leading `~` is expanded.
- `--set a.b.c=value` overrides any field after inheritance (`checkpoint=/path/to/step`,
  `guidance.w_lang=0.2`, `tr_log_layers=[13]`). Values are parsed as YAML.

## Fields

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
tr_log_layers: [2, 5, 10, 13, 15]
guidance:
  type: none                  # none | flat | tr_adaptive  (see docs/temporal_ratio.md)
```

`generate_video: false` stops the video denoising once the features are available and skips video
CFG, which is what the policy needs for control. `true` additionally finishes the schedule and
decodes the future frames (`future_video` in the policy output); because CFG changes the
denoising trajectory, the action head then sees slightly different features. A server request can
override it per query (`"generate_video": True` in the observation, see
[inference.md](inference.md#serving)).

`device` (default `cuda:0`) selects the GPU.
