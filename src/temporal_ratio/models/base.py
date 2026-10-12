import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch
from safetensors.torch import load_file

from temporal_ratio.models import action_head as _action_head
from temporal_ratio.shared import image_tools
import temporal_ratio.shared.array_typing as at

logger = logging.getLogger(__name__)

# Type variable for array types (JAX arrays, PyTorch tensors, or numpy arrays)
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


class ModelType(enum.Enum):
    """Supported model types."""
    VIDEO = "video_vla"


# This may need change if we release a small model.
IMAGE_RESOLUTION = (448, 448)


# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
# {
#     # Observation data.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB image in [-1, 1] or [0, 255]
#         ...  # Additional camera views
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True if image is valid
#         ...  # Masks for additional views
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt
#     "token_ar_mask": int32[*b, l],  # Optional, autoregressive mask for FAST model
#     "token_loss_mask": bool[*b, l],  # Optional, loss mask for FAST model
#
#      # Actions data.
#      "actions": float32[*b ah ad]
# }
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#
# @at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*b s"]

    wan_images_latents: at.Float[ArrayT, "*b wc wt wh ww"] | None = None
    wan_tokenized_prompt: at.Float[ArrayT, "*b w1 w2"] | None = None
    wan_tokenized_null_prompt: at.Float[ArrayT, "*b w1 w2"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")

        return cls(
            state=data["state"],
            wan_images_latents=data.get("wan_images_latents"),
            wan_tokenized_prompt=data.get("wan_tokenized_prompt"),
            wan_tokenized_null_prompt=data.get("wan_tokenized_null_prompt"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        return result


# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions = at.Float[ArrayT, "*b ah ad"]

def print_key_struct(pytree: at.PyTree, title: str) -> None:
    """Utility function to print the hierarchical key structure of a PyTree."""
    flat = traverse_util.flatten_dict(pytree)
    logger.info(f"{title}:")
    for key in flat:
        logger.info("  " + "/".join(key))

def prune_params_to_structure(expected: at.PyTree, params: at.PyTree) -> at.PyTree:
    """Return `params` with only leaves whose key paths exist in `expected`.

    This removes any extra parameters present in the hierarchical structure of
    `params` that don't have a corresponding leaf in `expected`.
    """
    expected_flat = traverse_util.flatten_dict(expected)
    params_flat = traverse_util.flatten_dict(params)
    filtered_flat = {kp: v for kp, v in params_flat.items() if kp in expected_flat}

    return traverse_util.unflatten_dict(filtered_flat)

def load_safetensors_partial(model, path, device="cpu"):
    state_dict = load_file(path, device=device)

    model_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []

    for k, v in model_state.items():
        target_k = k
        if target_k in state_dict:
            if state_dict[target_k].shape != v.shape:
                raise ValueError(
                    f"Shape mismatch for {k}: "
                    f"checkpoint {v.shape} vs model {model_state[k].shape} with key {target_k} and shape {state_dict[target_k].shape}"
                )
            model_state[k].copy_(state_dict[target_k])
            loaded_keys.append(k)
        else:
            skipped_keys.append(k)
            continue

    model.load_state_dict(model_state, strict=False)
    print("*************************** Model loading completed ***************************")
    print(f"Load path: {path}")
    print(f"Model class: {model.__class__.__name__}")
    print(f"Skipped keys: {skipped_keys}")
    print("************************************************************************")

    return loaded_keys, skipped_keys

@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str) -> "BaseModel":
        logger.info(f"train_config: {train_config}")
        # Infer video_feature_dim from checkpoint weights to avoid shape mismatch
        video_feature_dim = None
        metadata = safetensors.safe_open(weight_path, framework="pt")
        if "video_latent_embedder.weight" in metadata.keys():
            # embedder_in_dim = 2 * 2 * video_feature_dim
            shape = metadata.get_tensor("video_latent_embedder.weight").shape
            video_feature_dim = shape[1] // 4
            logger.info(f"Inferred video_feature_dim={video_feature_dim} from checkpoint")
        del metadata
        model = _action_head.VideoVLAPytorch(config=train_config.model, video_feature_dim=video_feature_dim)
        load_safetensors_partial(model, weight_path, device="cpu")
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
