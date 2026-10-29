# Checkpoints

A checkpoint is a step directory written by the trainers ([training.md](training.md#checkpoints)).
Inference needs `model.safetensors`, `metadata.pt`, `assets/**/norm_stats.json` and the DiT weights
(`video_model_full.pt`, or `lora_adapters/` applied to the pretrained Cosmos-Predict2.5 2B).

```bash
trvam download <repo> --subfolder <dir> --out checkpoints/      # or in a config:
#   checkpoint: hf://<repo>/<dir>
```

<!-- The LIBERO repository id is filled in when those checkpoints are published. -->

## LIBERO

Repository: `<LIBERO checkpoints repo>`. All use `policy_config: videoVLA_libero_short` (configs in
`configs/infer/libero/`) and read DiT block 19, except `*_L17` (block 17; picked up from the
checkpoint metadata).

| directory | training preset |
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

Repository: [`imishutk0410/TemporalRatio`](https://huggingface.co/imishutk0410/TemporalRatio)
(`checkpoint: hf://imishutk0410/TemporalRatio/<directory>`). The normalization statistics come from
the checkpoint.

| directory | content | config |
|---|---|---|
| `base` | multi-task policy for 24 bimanual tasks, 22 of which are in ABC-130k | `configs/infer/abc130k/yam_pp24.yaml` |
| `hybrid` | the multi-task action head with a separately fine-tuned full DiT | `configs/infer/abc130k/yam_pp24.yaml` |
| `task_finetunes/fold_tshirts` | `base` fine-tuned on T-shirt folding | `configs/infer/abc130k/fold_pen_bottle.yaml` with `policy_config=videoVLA_abc130k_folding_short` |
| `task_finetunes/pen_caps` | `base` fine-tuned on pen-cap opening | `configs/infer/abc130k/fold_pen_bottle.yaml` with `policy_config=videoVLA_abc130k_pen_caps_short` |
| `task_finetunes/bottle_caps` | `base` fine-tuned on bottle-cap unscrewing | `configs/infer/abc130k/fold_pen_bottle.yaml` with `policy_config=videoVLA_abc130k_assembly_short` |
| `longhorizon_video/goal_conditioned` | `base` with its DiT fine-tuned for goal-conditioned, long-horizon video of the three tasks above; `base`'s action head | `configs/infer/abc130k/fold_pen_bottle.yaml` |

Each directory holds the action head (`model.safetensors`), the DiT (`video_model_full.pt`, LoRA
merged in or fully fine-tuned), `metadata.pt` and the normalization statistics.

## Verified

Checked with this repository on one NVIDIA GB300:

- open-loop inference with and without video generation (`trvam infer`): LIBERO `joint_r32/25k`,
  `2stage_r32/100k`, `full_joint/85k` on `physical-intelligence/libero` episodes; YAM `878500`
  (merged DiT and base + LoRA adapters), `hybrid_fullvideo_dir150k_norm629k` and
  `hybrid_fullvideo_dir175k_norm629k` on converted ABC-130k episodes;
- TR-adaptive guidance, the websocket server and closed-loop LIBERO (`libero_spatial`, with and
  without guidance) with `joint_r32/25k`;
- fine-tuning `joint_r32/25k` and `hybrid_fullvideo_dir150k_norm629k`
  (`*/finetune_from_checkpoint.yaml`) and inference from the resulting checkpoints.
- fine-tuning a YAM checkpoint on single ABC-130k tasks (`folding`, `pen_caps`, `assembly`) and on
  the three together, continuing such a run for more epochs (`resume` with a larger step budget),
  and evaluating the results on whole validation episodes
  (`configs/infer/abc130k/fold_pen_bottle_heldout.yaml`);
- serving a fine-tuned checkpoint with `trvam serve` and querying it with
  `scripts/eval/check_server.py`, whose action errors match the open-loop evaluation of the same
  frames.

The other LIBERO checkpoints have the same format and configuration.
