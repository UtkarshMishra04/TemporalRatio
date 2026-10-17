"""Adaptive Action-level CFG with TR-modulated guidance weights.

At each replan step:
1. Run V2W with language → F_cond → action head with attention capture → a_cond + TR
2. Run V2W without language → F_uncond → action head → a_uncond
3. (Optional) Run V2W 93f with language → F_plan → action head → a_plan
4. Compute adaptive weight: w = w_base * clip(TR_k / TR_0 - 1, 0, max_scale)
5. Apply: a_final = a_cond + w * (a_cond - a_uncond) [+ w_plan_adaptive * (a_plan - a_cond)]

Key property:
  - ID: TR barely rises above TR0 (+6.6%) → w ≈ 0.3 → preserves SR
  - OOD: TR rises +18.3% → w ≈ 0.9 → boosts planning
  - Execution phase: TR drops below TR0 → w = 0 → pure policy

Usage:
    from temporal_ratio.guidance.tr_adaptive_policy import AdaptiveActionCFGWrapper
    wrapper = AdaptiveActionCFGWrapper(video_policy, w_base=5.0)
    wrapper.reset_episode()
    output = wrapper.infer(obs)  # tracks TR across replans
"""

import logging
import time
from typing import Any

import jax
import numpy as np
import torch
from openpi_client import base_policy as _base_policy
from PIL import Image
from typing_extensions import override

from temporal_ratio.models import base as _model
from temporal_ratio.guidance import scale as _gm
from temporal_ratio.guidance.tr import temporal_ratio as _temporal_ratio
from temporal_ratio.models.video_model import IMAGE_SIZES

log = logging.getLogger(__name__)

TOKENS_PER_LATENT_FRAME = 14 * 28  # 392 for LIBERO 224x448


def compute_tr_from_attention(action_model, observation, features, timestep,
                               noise, device, num_steps=1, tr_layer=None):
    """Run action head with attention capture, return actions + temporal ratio.

    Uses visualize_attention with num_steps=1 to capture attention from
    one denoising step.

    Args:
        tr_layer: Specific Gemma layer index to extract TR from.
                  None = average across all layers.
                  15 = most discriminative for LoRA 2Stage.
    """
    # Only hook the specific layer we need (or all if tr_layer is None)
    layers_to_hook = [tr_layer] if tr_layer is not None else None

    result = action_model.visualize_attention(
        device=device,
        observation=observation,
        video_latents=features,
        video_timestep=timestep,
        noise=noise,
        denoise_step_idx=0,
        num_steps=num_steps,
        save_path=None,
        layers=layers_to_hook,
        head_reduction="mean",
    )

    seg_attn = result["segment_attention"]  # (1, n_actions, n_segments) if single layer
    seg_names = result["segment_names"]

    video_indices = [i for i, name in enumerate(seg_names) if name.startswith("video_t")]
    if len(video_indices) < 2:
        return result["actions"].clone(), 1.0

    # seg_attn shape: (n_hooked_layers, n_actions, n_segments); use the first hooked layer.
    attn_per_seg = seg_attn.mean(axis=1)  # (n_hooked_layers, n_segments)
    tr = _temporal_ratio(attn_per_seg[0, video_indices])
    return result["actions"].clone(), tr


class AdaptiveActionCFGWrapper(_base_policy.BasePolicy):
    """Adaptive action-level CFG with TR-modulated guidance."""

    def __init__(
        self,
        video_policy,
        w_base_lang: float = 5.0,
        w_base_plan: float = 0.0,
        plan_latent_frames: int = 24,
        max_scale: float = 2.0,
        tr_layer: int = 2,  # Probe (2026-04-17): L=2 has max min-SNR across 8 ckpts.
        tr_baseline_mode: str = "per_episode",  # "per_episode" | "uncond_ratio" | "fixed"
        tr_baseline_fixed: float | None = None,  # used when mode == "fixed"
        cfg_formula: str = "extrapolation",  # "extrapolation" | "standard"
        tr_log_layers: list[int] | None = None,  # Additional layers to log TR for
        legacy_flat_floor: bool = False,  # True: reproduce original runs (scale 0 kept base-weight guidance)
        decode_video: bool = True,  # False: skip VAE decoding of the predicted future video
    ):
        self._vp = video_policy
        self.w_base_lang = w_base_lang
        self.w_base_plan = w_base_plan
        self.plan_latent_frames = plan_latent_frames
        self.max_scale = max_scale
        self.tr_layer = tr_layer
        self.cfg_formula = cfg_formula
        if tr_baseline_mode not in ("per_episode", "uncond_ratio", "fixed", "flat", "action_flat", "damped"):
            raise ValueError(f"unknown tr_baseline_mode={tr_baseline_mode}")
        if tr_baseline_mode == "fixed" and tr_baseline_fixed is None:
            raise ValueError("tr_baseline_mode='fixed' requires tr_baseline_fixed")
        self.tr_baseline_mode = tr_baseline_mode
        self.tr_baseline_fixed = tr_baseline_fixed
        self._tr_log_layers = tr_log_layers or []
        self.legacy_flat_floor = legacy_flat_floor
        self.decode_video = decode_video
        self.transforms = video_policy.transforms

        # Episode state
        self._tr_0 = None
        self._replan_count = 0
        self._tr_history = []  # [(tr_k, scale, w_lang, w_plan, tr_L*)] per replan

        log.info(f"AdaptiveActionCFG: w_base_lang={w_base_lang}, w_base_plan={w_base_plan}, "
                 f"max_scale={max_scale}, tr_layer={tr_layer}, "
                 f"baseline_mode={tr_baseline_mode}, tr_log_layers={self._tr_log_layers}")

    def reset_episode(self):
        """Call between episodes to reset TR baseline."""
        self._tr_0 = None
        self._replan_count = 0
        self._tr_history = []

    def _run_v2w(self, image_pil, prompt, num_video_frames, image_sizes):
        """Run V2W pipeline, return features + timestep + video latents."""
        vp = self._vp
        vel_pred, vel_ts, vid_lat = vp.video_model.pipe.eval_vid2world(
            prompt=prompt,
            initial_image=image_pil,
            num_video_frames=num_video_frames,
            resolution=f"{image_sizes[0]},{image_sizes[1]}",
            force_output_frames=True,
            intermediate_feature_ids=[vp.feature_id],
            num_steps=vp.num_global_denoising_steps,
            target_denoising_steps=None,
        )
        feat = vel_pred[vp.inference_denoising_steps][vp.feature_id].clone()
        ts = vp.head_timestep(vel_ts[vp.inference_denoising_steps])
        return feat, ts, vid_lat

    def _run_v2w_uncond(self, image_pil, num_video_frames, image_sizes):
        """Run V2W with zero text embedding."""
        vp = self._vp
        pipe = vp.video_model.pipe
        device = pipe.model_device

        uncond_key = ("__adaptive_uncond__", None)
        if uncond_key not in pipe._text_embedding_cache:
            zero_emb = torch.zeros(1, 512, 100352, dtype=torch.bfloat16, device=device)
            pipe._text_embedding_cache[uncond_key] = {"emb": zero_emb.cpu(), "neg_emb": zero_emb.cpu()}

        vel_pred, vel_ts, _ = pipe.eval_vid2world(
            prompt="__adaptive_uncond__",
            initial_image=image_pil,
            num_video_frames=num_video_frames,
            resolution=f"{image_sizes[0]},{image_sizes[1]}",
            force_output_frames=True,
            intermediate_feature_ids=[vp.feature_id],
            num_steps=vp.num_global_denoising_steps,
            target_denoising_steps=None,
        )
        feat = vel_pred[vp.inference_denoising_steps][vp.feature_id].clone()
        ts = vp.head_timestep(vel_ts[vp.inference_denoising_steps])
        return feat, ts

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None,
              num_latent_frames: int = 5) -> dict:
        vp = self._vp

        # Preprocessing
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = vp._input_transform(inputs)
        prompt = inputs["prompt"]
        inputs = vp._policy._remove_strings_transform(inputs)
        inputs = vp.make_wan_images(inputs)
        inputs = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x)).to(vp._policy._pytorch_device)[None, ...],
            inputs,
        )
        device = vp._policy._pytorch_device

        sample_kwargs = dict(vp._policy._sample_kwargs)
        if noise is not None:
            noise_t = torch.from_numpy(noise).to(device)
            if noise_t.ndim == 2:
                noise_t = noise_t[None, ...]
            sample_kwargs["noise"] = noise_t

        observation = _model.Observation.from_dict(inputs)
        image_sizes = IMAGE_SIZES[vp.vid2world_mode]
        num_video_frames = 4 * (num_latent_frames - 1) + 1
        VIDEO_LATENT_SHAPE = (num_latent_frames, image_sizes[0] // 16, image_sizes[1] // 16)

        vp._policy._model.set_video_latent_shape(VIDEO_LATENT_SHAPE)

        if "noise" not in sample_kwargs:
            action_model = vp._policy._model
            action_noise = action_model.sample_noise(
                (1, action_model.action_horizon, action_model.config.action_dim), device,
            )
            sample_kwargs["noise"] = action_noise

        combined_image = observation.wan_images_latents[0].cpu().numpy()
        combined_image_pil = Image.fromarray(combined_image.astype(np.uint8))

        # === 1. Conditioned V2W ===
        feat_cond, ts_cond, video_latents = self._run_v2w(
            combined_image_pil, prompt, num_video_frames, image_sizes
        )
        f_cond = feat_cond.view(1, VIDEO_LATENT_SHAPE[0], VIDEO_LATENT_SHAPE[1],
                                VIDEO_LATENT_SHAPE[2], vp.feature_dim)
        f_cond = f_cond[:, :num_latent_frames].reshape(1, -1, vp.feature_dim).clone()

        noise_clone = sample_kwargs["noise"].clone()

        # === 2. Unconditioned V2W (for lang CFG) ===
        f_uncond = None
        ts_uncond = None
        if self.w_base_lang > 0:
            feat_uncond, ts_uncond = self._run_v2w_uncond(
                combined_image_pil, num_video_frames, image_sizes
            )
            f_uncond = feat_uncond.view(1, VIDEO_LATENT_SHAPE[0], VIDEO_LATENT_SHAPE[1],
                                        VIDEO_LATENT_SHAPE[2], vp.feature_dim)
            f_uncond = f_uncond[:, :num_latent_frames].reshape(1, -1, vp.feature_dim).clone()

        # === 3. Plan V2W (for plan CFG) ===
        f_plan = None
        ts_plan = None
        if self.w_base_plan > 0:
            plan_frames = 4 * (self.plan_latent_frames - 1) + 1
            PLAN_SHAPE = (self.plan_latent_frames, image_sizes[0] // 16, image_sizes[1] // 16)
            feat_plan, ts_plan, _ = self._run_v2w(
                combined_image_pil, prompt, plan_frames, image_sizes
            )
            f_plan = feat_plan.view(1, PLAN_SHAPE[0], PLAN_SHAPE[1], PLAN_SHAPE[2], vp.feature_dim)
            f_plan = f_plan[:, :num_latent_frames].reshape(1, -1, vp.feature_dim).clone()

        # === 4. CFG ===
        action_model = vp._policy._model
        num_steps = sample_kwargs.get("num_steps", 10)

        if self.tr_baseline_mode == "action_flat":
            # Action-level flat CFG: denoise each feature set independently, interpolate final actions
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                a_cond = action_model.sample_actions(
                    device, observation, f_cond, ts_cond,
                    noise=noise_clone.clone(), num_steps=num_steps,
                )
                if f_uncond is not None:
                    a_uncond = action_model.sample_actions(
                        device, observation, f_uncond, ts_uncond,
                        noise=noise_clone.clone(), num_steps=num_steps,
                    )
                if f_plan is not None:
                    a_plan = action_model.sample_actions(
                        device, observation, f_plan, ts_plan,
                        noise=noise_clone.clone(), num_steps=num_steps,
                    )

            a_guided = a_cond.clone()
            if f_uncond is not None:
                a_guided = a_guided + self.w_base_lang * (a_cond - a_uncond)
            if f_plan is not None:
                a_guided = a_guided + self.w_base_plan * (a_plan - a_cond)

            tr_k = 0.0
            w_lang_eff = self.w_base_lang
            w_plan_eff = self.w_base_plan
            raw_scale = 1.0
            scale = 1.0
            if self._tr_0 is None:
                self._tr_0 = 0.0
        else:
            # Velocity-space CFG
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                a_guided, tr_k = action_model.sample_actions_adaptive_cfg(
                    device, observation,
                    video_latents_cond=f_cond, video_timestep_cond=ts_cond,
                    video_latents_uncond=f_uncond, video_timestep_uncond=ts_uncond,
                    video_latents_plan=f_plan, video_timestep_plan=ts_plan,
                    w_lang=self.w_base_lang,
                    w_plan=self.w_base_plan,
                    noise=noise_clone.clone(),
                    num_steps=num_steps,
                    tr_layer=self.tr_layer,
                    cfg_formula=self.cfg_formula,
                )

            if self.tr_baseline_mode == "flat":
                w_lang_eff = self.w_base_lang
                w_plan_eff = self.w_base_plan
                raw_scale = 1.0
                scale = 1.0
                if self._tr_0 is None:
                    self._tr_0 = tr_k
            elif self.tr_baseline_mode == "damped":
                if self._tr_0 is None:
                    self._tr_0 = tr_k
                raw_scale, scale = _gm.damped_scale(tr_k, self._tr_0, self.max_scale)
                w_lang_eff = self.w_base_lang * scale
                w_plan_eff = self.w_base_plan * scale

                if w_lang_eff > 0.01 or w_plan_eff > 0.01:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        a_guided, _ = action_model.sample_actions_adaptive_cfg(
                            device, observation,
                            video_latents_cond=f_cond, video_timestep_cond=ts_cond,
                            video_latents_uncond=f_uncond, video_timestep_uncond=ts_uncond,
                            video_latents_plan=f_plan, video_timestep_plan=ts_plan,
                            w_lang=w_lang_eff,
                            w_plan=w_plan_eff,
                            noise=noise_clone.clone(),
                            num_steps=num_steps,
                            tr_layer=self.tr_layer,
                            cfg_formula=self.cfg_formula,
                        )
            else:
                if self.tr_baseline_mode == "per_episode":
                    if self._tr_0 is None:
                        self._tr_0 = tr_k
                    tr_baseline = self._tr_0
                elif self.tr_baseline_mode == "fixed":
                    tr_baseline = self.tr_baseline_fixed
                else:  # uncond_ratio
                    if f_uncond is not None:
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            _, tr_uncond_val = action_model.sample_actions_adaptive_cfg(
                                device, observation,
                                video_latents_cond=f_uncond, video_timestep_cond=ts_uncond,
                                w_lang=0.0, w_plan=0.0,
                                noise=noise_clone.clone(), num_steps=num_steps,
                                tr_layer=self.tr_layer,
                            )
                        tr_baseline = tr_uncond_val
                    else:
                        tr_baseline = tr_k
                    if self._tr_0 is None:
                        self._tr_0 = tr_k

                raw_scale, scale = _gm.adaptive_scale(tr_k, tr_baseline, self.max_scale)
                decision = _gm.decide(
                    scale, self.w_base_lang, self.w_base_plan, legacy_flat_floor=self.legacy_flat_floor
                )
                w_lang_eff, w_plan_eff = decision.w_lang, decision.w_plan

                if decision.resample:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        a_guided, _ = action_model.sample_actions_adaptive_cfg(
                            device, observation,
                            video_latents_cond=f_cond, video_timestep_cond=ts_cond,
                            video_latents_uncond=f_uncond, video_timestep_uncond=ts_uncond,
                            video_latents_plan=f_plan, video_timestep_plan=ts_plan,
                            w_lang=w_lang_eff,
                            w_plan=w_plan_eff,
                            noise=noise_clone.clone(),
                            num_steps=num_steps,
                            tr_layer=self.tr_layer,
                            cfg_formula=self.cfg_formula,
                        )

        feat_cos = 1.0
        if f_uncond is not None:
            feat_cos = torch.nn.functional.cosine_similarity(
                f_cond.reshape(1, -1), f_uncond.reshape(1, -1)
            ).item()

        # Decode the predicted future video (optional; costs one VAE decode per replan)
        video_np = None
        if self.decode_video:
            video_prediction = vp.video_model.pipe.model.decode(video_latents).float()
            video = video_prediction.squeeze(0).permute(1, 2, 3, 0)
            video = ((video + 1) / 2).clamp(0, 1) * 255
            video_np = video.cpu().numpy().astype(np.uint8)

        # Output
        outputs = {"state": inputs["state"], "actions": a_guided}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        outputs = vp._policy._output_transform(outputs)
        outputs["future_video"] = video_np
        outputs["features"] = f_cond[0].detach().cpu().numpy()
        outputs["adaptive_cfg_info"] = {
            "tr_k": tr_k,
            "tr_0": self._tr_0,
            "tr_baseline_mode": self.tr_baseline_mode,
            "raw_scale": raw_scale,
            "w_lang": w_lang_eff,
            "w_plan": w_plan_eff,
            "feat_cos": feat_cos,
            "replan_idx": self._replan_count,
        }

        tr_entry = {
            "tr_k": tr_k, "scale": raw_scale, "w_lang": w_lang_eff, "w_plan": w_plan_eff,
        }

        # Multi-layer TR logging (if configured)
        if hasattr(self, '_tr_log_layers') and self._tr_log_layers:
            for layer in self._tr_log_layers:
                if layer == self.tr_layer:
                    tr_entry[f"tr_L{layer}"] = tr_k
                else:
                    _, tr_val = compute_tr_from_attention(
                        action_model, observation, f_cond, ts_cond,
                        noise_clone.clone(), device, num_steps=1, tr_layer=layer,
                    )
                    tr_entry[f"tr_L{layer}"] = tr_val

        self._tr_history.append(tr_entry)
        self._replan_count += 1
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return getattr(self._vp, "metadata", {})
