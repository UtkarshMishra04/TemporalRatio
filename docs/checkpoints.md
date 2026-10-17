# Checkpoints

A checkpoint is a step directory written by the trainers (on the `master` branch).
Inference needs `model.safetensors`, `metadata.pt`, `assets/**/norm_stats.json` and the DiT weights
(`video_model_full.pt`, or `lora_adapters/` applied to the pretrained Cosmos-Predict2.5 2B).

```bash
trvam download <repo> --subfolder <dir> --out checkpoints/      # or in a config:
#   checkpoint: hf://<repo>/<dir>
```

<!-- Repository ids are filled in when the checkpoints are published. -->

## LIBERO

Repository: `<LIBERO checkpoints repo>`. All use `policy_config: videoVLA_libero_short` (configs in
`configs/infer/libero/`) and read DiT block 19, except `*_L17` (block 17; picked up from the
checkpoint metadata).

| directory | training preset (`configs/train/` on `master`) |
|---|---|
| `checkpoints/joint_r16/{25k,100k}` | `libero/joint_lora_r16` |
| `checkpoints/joint_r32/{25k,100k}` | `libero/joint_lora_r32` |
| `checkpoints/joint_r64/100k` | `libero/joint_lora_r64` |
| `checkpoints/full_joint/85k` | `libero/joint_full` |
| `checkpoints/joint_93f_r{16,32,64}/100k` | `libero/joint_93f_lora_r{16,32,64}` |
| `checkpoints/full_joint_93f/100k` | `libero/joint_93f_full` |
| `checkpoints/2stage_r{16,32,64}/100k` | `libero/two_stage_lora_r{16,32,64}` (stage 2 step) |
| `checkpoints/full_2stage/{50k,100k}` | `libero/two_stage_full` |
| `checkpoints/full_2stage_L17/100k` | `libero/two_stage_full` with `model.feature_layer=17` |
| `checkpoints/2stage_93f_r{16,32,64}/100k` | `libero/two_stage_93f_lora_r32` (rank 16 / 32 / 64) |
| `checkpoints/full_2stage_93f/{75k,100k}` | `libero/two_stage_93f_full` |
| `checkpoints/full_2stage_93f_L17/100k` | `libero/two_stage_93f_full` with `model.feature_layer=17` |

The step directories contain the LoRA-merged DiT (`video_model_full.pt`) and the action head; the
LIBERO normalization statistics are identical for all of them (and ship in `assets/`).

## Bimanual YAM (ABC-130k)

Repository: `<YAM checkpoints repo>`. Use `configs/infer/abc130k/yam_pp24.yaml`
(`policy_config: videoVLA_abc130k_pp24_short`); the normalization statistics come from the
checkpoint. Trained on 24 bimanual tasks, 22 of which are in ABC-130k.

| directory | content |
|---|---|
| `878500` | step 878,500 of the 24-task joint run (`abc130k/pp24_joint_lora_r32`): action head, `lora_adapters/` and the merged `video_model_full.pt` (both give identical outputs), EMA and optimizer state |
| `hybrid_fullvideo_dir150k_norm629k` | action head of the same run (step 629,500) with a separately fine-tuned full DiT (`video_model_full.pt`) |
| `hybrid_fullvideo_dir175k_norm629k` | the same action head with another full-DiT checkpoint |

## Verified

Checked on one NVIDIA GB300 with the `master` branch, whose inference code this branch shares:

- inference with and without video generation: LIBERO `joint_r32/25k`, `2stage_r32/100k`,
  `full_joint/85k`; YAM `878500` (merged DiT and base + LoRA adapters),
  `hybrid_fullvideo_dir150k_norm629k` and `hybrid_fullvideo_dir175k_norm629k`;
- TR-adaptive guidance and the websocket server with `joint_r32/25k`;
- serving a fine-tuned YAM checkpoint with `trvam serve` to a websocket client.

The other LIBERO checkpoints have the same format and configuration.
