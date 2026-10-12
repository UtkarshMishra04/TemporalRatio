import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from temporal_ratio.models import base as _model
import temporal_ratio.models.gemma as _gemma
from temporal_ratio.shared import array_typing as at
import temporal_ratio.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from temporal_ratio.models.action_head import VideoVLAPytorch


@dataclasses.dataclass(frozen=True)
class VideoVLAConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    state_dim: int = 32
    observation_horizon: int = 50
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    interleaved: bool = False
    multimodal: bool = False

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", True)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.VIDEO

    @override
    def create(self, rng: at.KeyArrayLike) -> "VideoVLAPytorch":
        from temporal_ratio.models.action_head import VideoVLAPytorch

        return VideoVLAPytorch(self)

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        
        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32)
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)

    def set_action_expert_config(self, action_expert_variant: _gemma.Variant) -> "VideoVLAConfig":
        """Set action_expert_config in the model based on the data config."""
        return dataclasses.replace(self, action_expert_variant=action_expert_variant)
