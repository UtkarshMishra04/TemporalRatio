"""Action-level Classifier-Free Guidance.

Runs the video model twice (with and without language), feeds both feature
sets through the action head, then applies CFG in action space:

    a_final = a_uncond + guidance_weight * (a_cond - a_uncond)

Key property: self-adaptive
  - ID scenes: F_cond ≈ F_uncond → guidance barely changes actions → preserves ID SR
  - OOD scenes: F_cond ≠ F_uncond → guidance amplifies language signal → improves OOD SR

Usage:
    from temporal_ratio.guidance.action_cfg_policy import ActionCFGWrapper

    wrapper = ActionCFGWrapper(video_policy, guidance_weight=2.0)
    output = wrapper.infer(obs)
"""

import logging
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from openpi_client import base_policy as _base_policy
from PIL import Image
from typing_extensions import override

from temporal_ratio.models import base as _model
from temporal_ratio.models.video_model import IMAGE_SIZES

log = logging.getLogger(__name__)


class ActionCFGWrapper(_base_policy.BasePolicy):
    """Wraps VideoModelPolicyWrapper with action-level classifier-free guidance.

    Supports two guidance terms:
      1. Language CFG: a_policy + w_lang * (a_policy - a_no_lang)
      2. Planning CFG: + w_plan * (a_plan_93f - a_policy)

    Set w_plan=0 for pure language CFG (2 V2W passes).
    Set both >0 for full plan+language CFG (3 V2W passes).
    """

    def __init__(
        self,
        video_policy,  # VideoModelPolicyWrapper instance
        guidance_weight: float = 2.0,      # w_lang: language guidance
        plan_weight: float = 0.0,           # w_plan: planning guidance (93f)
        plan_latent_frames: int = 24,       # latent frames for planning (24=93f)
    ):
        self._vp = video_policy
        self.guidance_weight = guidance_weight
        self.plan_weight = plan_weight
        self.plan_latent_frames = plan_latent_frames
        self.transforms = video_policy.transforms

        log.info(f"ActionCFGWrapper: w_lang={guidance_weight}, w_plan={plan_weight}, "
                 f"plan_lat={plan_latent_frames}")

    def _run_standard_v2w(self, combined_image, prompt, num_video_frames, image_sizes):
        """Run full V2W pipeline (with video-level CFG). Returns features + latents."""
        vp = self._vp
        velocity_prediction, velocity_timestep, video_latents = vp.video_model.pipe.eval_vid2world(
            prompt=prompt,
            initial_image=combined_image,
            num_video_frames=num_video_frames,
            resolution=f"{image_sizes[0]},{image_sizes[1]}",
            force_output_frames=True,
            intermediate_feature_ids=[vp.feature_id],
            num_steps=vp.num_global_denoising_steps,
            target_denoising_steps=None,
        )
        feat = velocity_prediction[vp.inference_denoising_steps][vp.feature_id].clone()
        ts = velocity_timestep[vp.inference_denoising_steps].unsqueeze(0).to(torch.float32) / 1000.0
        return feat, ts, video_latents

    def _extract_uncond_features(self, combined_image_pil, num_video_frames, image_sizes):
        """Extract features without language by pre-caching a zero embedding.

        Uses the same eval_vid2world path but with a zero text embedding
        injected into the text embedding cache.
        """
        vp = self._vp
        pipe = vp.video_model.pipe
        device = pipe.model_device

        # Cache a zero embedding as the prompt so eval_vid2world uses it
        zero_emb = torch.zeros(1, 512, 100352, dtype=torch.bfloat16, device=device)
        # Temporarily inject zero embedding for the empty prompt
        _UNCOND_PROMPT = "__uncond__"
        pipe._text_embedding_cache[(_UNCOND_PROMPT, None)] = {
            "emb": zero_emb.cpu(),
            "neg_emb": zero_emb.cpu(),
        }

        velocity_prediction, velocity_timestep, video_latents = pipe.eval_vid2world(
            prompt=_UNCOND_PROMPT,
            initial_image=combined_image_pil,
            num_video_frames=num_video_frames,
            resolution=f"{image_sizes[0]},{image_sizes[1]}",
            force_output_frames=True,
            intermediate_feature_ids=[vp.feature_id],
            num_steps=vp.num_global_denoising_steps,
            target_denoising_steps=None,
        )
        feat = velocity_prediction[vp.inference_denoising_steps][vp.feature_id].clone()
        ts = velocity_timestep[vp.inference_denoising_steps].unsqueeze(0).to(torch.float32) / 1000.0

        # Clean up cache entry
        del pipe._text_embedding_cache[(_UNCOND_PROMPT, None)]

        return feat, ts

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None,
              num_latent_frames: int = 5) -> dict:
        vp = self._vp

        # === Preprocessing (same as VideoModelPolicyWrapper.infer) ===
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = vp._input_transform(inputs)
        prompt = inputs['prompt']
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
        ACTION_LATENT_SHAPE = VIDEO_LATENT_SHAPE
        vp._policy._model.set_video_latent_shape(ACTION_LATENT_SHAPE)

        # Pre-generate shared action noise
        if "noise" not in sample_kwargs:
            action_model = vp._policy._model
            action_noise = action_model.sample_noise(
                (1, action_model.action_horizon, action_model.config.action_dim),
                device,
            )
            sample_kwargs["noise"] = action_noise

        combined_image = observation.wan_images_latents[0].cpu().numpy()
        combined_image_pil = Image.fromarray(combined_image.astype(np.uint8))

        # === Run V2W with language (conditioned) — full pipeline ===
        feat_cond, ts_cond, video_latents = self._run_standard_v2w(
            combined_image_pil, prompt, num_video_frames, image_sizes
        )

        # === Extract features without language (zero text embedding) ===
        feat_uncond, ts_uncond = self._extract_uncond_features(
            combined_image_pil, num_video_frames, image_sizes
        )

        # Reshape and crop features
        def prepare_features(feat, src_latent_shape):
            f = feat.view(1, src_latent_shape[0], src_latent_shape[1],
                         src_latent_shape[2], vp.feature_dim)
            f = f[:, :num_latent_frames, :, :, :]
            return f.view(1, -1, vp.feature_dim).clone()

        f_cond = prepare_features(feat_cond, VIDEO_LATENT_SHAPE)
        f_uncond = prepare_features(feat_uncond, VIDEO_LATENT_SHAPE)

        # === Optionally run 93f planning pass ===
        f_plan = None
        if self.plan_weight > 0:
            plan_frames = 4 * (self.plan_latent_frames - 1) + 1
            PLAN_LATENT_SHAPE = (self.plan_latent_frames, image_sizes[0] // 16, image_sizes[1] // 16)
            feat_plan, ts_plan, _ = self._run_standard_v2w(
                combined_image_pil, prompt, plan_frames, image_sizes
            )
            f_plan = prepare_features(feat_plan, PLAN_LATENT_SHAPE)

        # === Run action head (clone outputs to avoid CUDA graph overwrites) ===
        noise_clone = sample_kwargs["noise"].clone()

        a_cond = vp._policy._sample_actions(
            device, observation, f_cond, ts_cond,
            **{**sample_kwargs, "noise": noise_clone.clone()}
        ).clone()
        a_uncond = vp._policy._sample_actions(
            device, observation, f_uncond, ts_uncond,
            **{**sample_kwargs, "noise": noise_clone.clone()}
        ).clone()

        # === CFG in action space ===
        # Language guidance: amplify what language adds
        a_guided = a_cond + self.guidance_weight * (a_cond - a_uncond)

        # Planning guidance: amplify what 93f goal features add
        plan_delta = 0.0
        if f_plan is not None:
            a_plan = vp._policy._sample_actions(
                device, observation, f_plan, ts_plan,
                **{**sample_kwargs, "noise": noise_clone.clone()}
            ).clone()
            a_guided = a_guided + self.plan_weight * (a_plan - a_cond)
            plan_delta = (a_plan - a_cond).abs().mean().item()

        # Compute monitoring stats
        with torch.no_grad():
            lang_delta = (a_cond - a_uncond).abs().mean().item()
            feature_cos = torch.nn.functional.cosine_similarity(
                f_cond.reshape(1, -1), f_uncond.reshape(1, -1)
            ).item()

        # Decode video from conditioned latents
        video_prediction = vp.video_model.pipe.model.decode(video_latents).float()
        video = video_prediction.squeeze(0).permute(1, 2, 3, 0)
        video = ((video + 1) / 2).clamp(0, 1) * 255
        video_np = video.cpu().numpy().astype(np.uint8)

        # === Output ===
        outputs = {
            "state": inputs["state"],
            "actions": a_guided,
        }
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        outputs = vp._policy._output_transform(outputs)
        outputs["future_video"] = video_np
        outputs["features"] = f_cond[0].detach().cpu().numpy()
        outputs["cfg_info"] = {
            "w_lang": self.guidance_weight,
            "w_plan": self.plan_weight,
            "lang_delta": lang_delta,
            "plan_delta": plan_delta,
            "feature_cos_sim": feature_cos,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return getattr(self._vp, "metadata", {})
