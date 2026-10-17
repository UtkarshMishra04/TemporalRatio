"""Inference policies: Gemma action head (``Policy``) driven by Cosmos video features (``VideoModelPolicyWrapper``).

One ``VideoModelPolicyWrapper.infer`` call:

1. applies the robot's input transforms (repack, normalization, resizing) and tiles the cameras
   into one frame (``VIDEO_SETUPS[setup].camera_keys``, left to right);
2. runs the Cosmos-Predict2.5 Video2World pipeline conditioned on that frame and the prompt and
   reads the DiT features of block ``feature_id`` after ``inference_denoising_steps`` of
   ``num_global_denoising_steps`` video denoising steps;
3. samples an action chunk from the action head conditioned on those features.

``generate_video=False`` (action-only) stops the video denoising at the feature step and skips
video CFG; ``generate_video=True`` runs the full schedule with video CFG (``video_guidance``) and
decodes the predicted future frames. Because CFG changes the denoising trajectory, the two modes
feed slightly different features to the action head.
"""

from __future__ import annotations

from collections.abc import Sequence
import logging
import time
from typing import Any, TypeAlias

import jax
import numpy as np
from openpi_client import base_policy as _base_policy
from PIL import Image
import torch
from typing_extensions import override

from temporal_ratio.data import transforms as _transforms
from temporal_ratio.models import base as _model
from temporal_ratio.models.video_model import VIDEO_SETUPS, VideoProcessor

BasePolicy: TypeAlias = _base_policy.BasePolicy

log = logging.getLogger(__name__)


def head_timestep(timestep: torch.Tensor, train_timesteps: torch.Tensor) -> torch.Tensor:
    """Video timestep for the action head, shape ``(1,)``, from a sampling-schedule timestep (0-1000).

    The action heads are trained on the fraction ``u`` in [0, 1) that the DiT's training scheduler
    maps to the noise level of the features (``RectifiedFlow.get_discrete_timestamp``:
    ``timesteps[int(u * N)]``), so inference returns ``index / N`` of the training timestep nearest
    to ``timestep`` (e.g. u = 0.336 for sigma 0.908 with shift 5).
    """
    t = timestep.reshape(1).to(torch.float32)
    grid = train_timesteps.to(device=t.device, dtype=torch.float32)
    index = torch.argmin((grid - t).abs())
    return (index.to(torch.float32) / len(grid)).reshape(1)


def _to_numpy(outputs: dict) -> dict:
    return jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)


class Policy(BasePolicy):
    """The action head with its input/output transforms (no video model)."""

    def __init__(
        self,
        model: _model.BaseModel,
        *,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = True,
    ):
        if not is_pytorch:
            raise ValueError("only PyTorch checkpoints are supported")
        self._model = model.to(pytorch_device)
        self._model.eval()
        self.transforms = transforms
        self._input_transform = _transforms.compose(transforms)
        self._remove_strings_transform = _transforms.RemovePrompt()
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = True
        self._pytorch_device = pytorch_device
        self._sample_actions = model.sample_actions

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        raise NotImplementedError("the action head needs video features; use VideoModelPolicyWrapper")

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class VideoModelPolicyWrapper(BasePolicy):
    """Cosmos video model + action head policy."""

    def __init__(
        self,
        policy: Policy,
        device: str = "cpu",
        inference_denoising_steps: int = 0,
        vid2world_mode: str = "libero_flat_short",
        is_causal: bool = False,
        feature_id: int = 19,
        num_global_denoising_steps: int = 35,
        vid2world_ckpt: str | None = None,
        video_guidance: float = 7.0,
    ):
        """
        Args:
            policy: action head + transforms, from ``policies.factory.create_trained_policy``.
            inference_denoising_steps: video denoising step whose DiT features feed the action head.
            vid2world_mode: video setup name (``models.video_model.VIDEO_SETUPS``).
            feature_id: DiT block whose output feeds the action head (19 for the released models).
            num_global_denoising_steps: length of the video denoising schedule.
            vid2world_ckpt: fine-tuned DiT (``video_model_full.pt`` or a checkpoint directory with
                ``video_model_full.pt`` / ``lora_adapters/``); ``None`` = pretrained base.
            video_guidance: video classifier-free guidance, used only when generating video.
        """
        if vid2world_mode not in VIDEO_SETUPS:
            raise ValueError(f"unknown video setup {vid2world_mode!r}; choose from {sorted(VIDEO_SETUPS)}")
        self._policy = policy
        self.vid2world_mode = vid2world_mode
        self.setup = VIDEO_SETUPS[vid2world_mode]
        self.is_causal = is_causal
        self.feature_id = feature_id
        self.feature_dim = self.setup.feature_dim
        self.num_global_denoising_steps = num_global_denoising_steps
        self.inference_denoising_steps = inference_denoising_steps
        self.video_guidance = video_guidance
        if not 0 <= inference_denoising_steps < num_global_denoising_steps:
            raise ValueError("inference_denoising_steps must be in [0, num_global_denoising_steps)")
        self.video_model = VideoProcessor(
            setup=vid2world_mode,
            device=device,
            atten_temporal_causal=is_causal,
            num_global_denoising_steps=num_global_denoising_steps,
            ckpt_path=vid2world_ckpt,
        )
        self.transforms = list(self._policy.transforms)
        self._input_transform = _transforms.compose(self.transforms)
        self.device = device

    def head_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        """The action head's video timestep for features at sampling timestep ``timestep``."""
        train_timesteps = self.video_model.pipe.model.rectified_flow.noise_scheduler.timesteps
        return head_timestep(timestep, train_timesteps)

    # ------------------------------------------------------------------ inputs
    def make_wan_images(self, inputs: dict) -> dict:
        """Tile the setup's cameras left to right into ``inputs["wan_images_latents"]``."""
        images = inputs.pop("image")
        frames = [np.asarray(images[key]) for key in self.setup.camera_keys]
        inputs["wan_images_latents"] = np.concatenate(frames, axis=1).astype(np.float64)
        return inputs

    def _prepare(self, obs: dict, noise: np.ndarray | None):
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        prompt = inputs["prompt"]
        inputs = self._policy._remove_strings_transform(inputs)
        inputs = self.make_wan_images(inputs)
        device = self._policy._pytorch_device
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)

        sample_kwargs = dict(self._policy._sample_kwargs)
        action_model = self._policy._model
        if noise is not None:
            noise_t = torch.from_numpy(noise).to(device)
            sample_kwargs["noise"] = noise_t[None, ...] if noise_t.ndim == 2 else noise_t
        else:
            # Draw the action noise before the DiT runs (CUDA-graph replays restore the RNG state).
            sample_kwargs["noise"] = action_model.sample_noise(
                (1, action_model.action_horizon, action_model.config.action_dim), device
            )
        observation = _model.Observation.from_dict(inputs)
        frame = Image.fromarray(observation.wan_images_latents[0].cpu().numpy().astype(np.uint8))
        return inputs, prompt, observation, frame, sample_kwargs

    def _video_features(self, prompt, frame, *, num_latent_frames, num_gen_latent_frames, generate_video):
        """Run the video model; return (features, timestep, decoded video or None)."""
        h, w = self.setup.image_size
        gen_latent = num_gen_latent_frames or num_latent_frames
        latent_shape = (gen_latent, h // 16, w // 16)
        action_only = not generate_video
        velocity, timesteps, video_latents = self.video_model.pipe.eval_vid2world(
            prompt=prompt,
            initial_image=frame,
            guidance=self.video_guidance,
            num_video_frames=4 * (gen_latent - 1) + 1,
            resolution=f"{h},{w}",
            force_output_frames=True,
            intermediate_feature_ids=[self.feature_id],
            num_steps=self.num_global_denoising_steps,
            target_denoising_steps=self.inference_denoising_steps if action_only else None,
            skip_cfg=action_only,
        )
        video_np = None
        if generate_video:
            video = self.video_model.pipe.model.decode(video_latents).float()
            video = video.squeeze(0).permute(1, 2, 3, 0)
            video_np = (((video + 1) / 2).clamp(0, 1) * 255).cpu().numpy().astype(np.uint8)

        feature = velocity[self.inference_denoising_steps][self.feature_id].clone()
        timestep = self.head_timestep(timesteps[self.inference_denoising_steps])
        expected = latent_shape[0] * latent_shape[1] * latent_shape[2]
        if feature.shape[1] != expected:
            raise RuntimeError(f"expected {expected} feature tokens, got {tuple(feature.shape)}")
        feature = feature.view(1, *latent_shape, self.feature_dim)[:, :num_latent_frames]
        return feature.reshape(1, -1, self.feature_dim).clone(), timestep, video_np

    # ------------------------------------------------------------------ inference
    @override
    def infer(  # type: ignore[override]
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        num_latent_frames: int = 5,
        num_gen_latent_frames: int | None = None,
        generate_video: bool = True,
        tr_layers: Sequence[int] | None = None,
    ) -> dict:
        """Predict an action chunk.

        Args:
            obs: raw observation dict (robot keys, see ``policies/*_policy.py``) with a ``prompt``.
            noise: optional initial action noise ``(action_horizon, action_dim)``.
            num_latent_frames: latent frames the action head reads (5 = 17 video frames).
            num_gen_latent_frames: generate a longer video than the action head reads.
            generate_video: finish video denoising (with CFG) and decode the predicted frames.
            tr_layers: also return the Temporal Ratio of these action-expert layers, computed on
                the features fed to the action head.

        Returns ``actions`` (action_horizon, robot action dim), ``future_video`` (T, H, W, 3) uint8
        or ``None``, ``features``, ``policy_timing`` and, if requested, ``temporal_ratio``.
        """
        inputs, prompt, observation, frame, sample_kwargs = self._prepare(obs, noise)
        h, w = self.setup.image_size
        action_model = self._policy._model
        action_model.set_video_latent_shape((num_latent_frames, h // 16, w // 16))
        device = self._policy._pytorch_device

        features, timestep, video_np = self._video_features(
            prompt, frame, num_latent_frames=num_latent_frames,
            num_gen_latent_frames=num_gen_latent_frames, generate_video=generate_video,
        )
        start = time.monotonic()
        actions = self._policy._sample_actions(device, observation, features, timestep, **sample_kwargs)
        head_time = time.monotonic() - start

        temporal_ratio = None
        if tr_layers:
            # Attention weights are only materialized by the eager kernel; restore the previous
            # implementation afterwards so later action sampling is unaffected.
            expert_config = action_model.paligemma_with_expert.gemma_expert.config
            attn_impl = expert_config._attn_implementation
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
                    _, per_layer = action_model.sample_actions_adaptive_cfg(
                        device, observation, video_latents_cond=features, video_timestep_cond=timestep,
                        w_lang=0.0, w_plan=0.0, noise=sample_kwargs["noise"].clone(),
                        num_steps=sample_kwargs.get("num_steps", 10), tr_all_layers=True,
                    )
            finally:
                expert_config._attn_implementation = attn_impl
            temporal_ratio = {int(layer): float(per_layer[layer]) for layer in tr_layers if layer in per_layer}

        outputs = _to_numpy({"state": inputs["state"], "actions": actions})
        outputs = self._policy._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": head_time * 1000}
        outputs["future_video"] = video_np
        outputs["features"] = features[0].detach().cpu().numpy()
        if temporal_ratio is not None:
            outputs["temporal_ratio"] = temporal_ratio
        return outputs

    def infer_with_attention(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        num_latent_frames: int = 5,
        save_path: str | None = None,
        denoise_step_idx: int = 0,
        layers: list[int] | None = None,
        head_reduction: str = "mean",
        generate_video: bool = True,
    ) -> dict:
        """``infer`` that also returns the action expert's attention over video/state/action segments.

        Returns the ``infer`` outputs plus ``attention_info`` with ``segment_attention`` and
        ``segment_names`` (and saves a heatmap to ``save_path`` if given).
        """
        inputs, prompt, observation, frame, sample_kwargs = self._prepare(obs, noise)
        h, w = self.setup.image_size
        self._policy._model.set_video_latent_shape((num_latent_frames, h // 16, w // 16))
        features, timestep, video_np = self._video_features(
            prompt, frame, num_latent_frames=num_latent_frames, num_gen_latent_frames=None,
            generate_video=generate_video,
        )
        attn = self._policy._model.visualize_attention(
            device=self._policy._pytorch_device,
            observation=observation,
            video_latents=features,
            video_timestep=timestep,
            denoise_step_idx=denoise_step_idx,
            num_steps=sample_kwargs.get("num_steps", 10),
            save_path=save_path,
            layers=layers,
            head_reduction=head_reduction,
            **{k: v for k, v in sample_kwargs.items() if k != "num_steps"},
        )
        outputs = _to_numpy({"state": inputs["state"], "actions": attn["actions"]})
        outputs = self._policy._output_transform(outputs)
        outputs["future_video"] = video_np
        outputs["attention_info"] = {
            "segment_attention": attn["segment_attention"],
            "segment_names": attn["segment_names"],
        }
        return outputs
