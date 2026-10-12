import dataclasses

import einops
import numpy as np

from temporal_ratio.data import transforms
from temporal_ratio.models import base as _model


def make_yam_example() -> dict:
    """Creates a random input example for the YAM policy."""
    return {
        "observation/state": np.random.rand(14),
        "observation/left_image": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "observation/left_wrist_left": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "observation/right_wrist_left": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)

    if image.ndim == 4:
        if image.shape[1] == 3:
            image = einops.rearrange(image, "b c h w -> b h w c")
    else:
        if image.shape[0] == 3:
            image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class YAMInputs(transforms.DataTransformFn):
    """Converts YAM dataset inputs to model format.

    YAM has 3 cameras: left_image (base), left_wrist_left, right_wrist_left.
    We map them to robocasa-compatible keys so WanTransformedDataset handles them.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        left_image = _parse_image(data["observation/left_image"])
        left_wrist_left = _parse_image(data["observation/left_wrist_left"])
        right_wrist_left = _parse_image(data["observation/right_wrist_left"])

        # Map to robocasa-compatible image keys for WanTransformedDataset:
        # left_base_0_rgb = left_image (base camera)
        # right_base_0_rgb = left_wrist_left (second view)
        # left_wrist_0_rgb = right_wrist_left (third view)
        # This preserves the [left_image | left_wrist_left | right_wrist_left] horizontal order
        # matching the video world model training layout.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "left_base_0_rgb": left_image,
                "right_base_0_rgb": left_wrist_left,
                "left_wrist_0_rgb": right_wrist_left,
                "right_wrist_0_rgb": np.zeros_like(left_image),
            },
            "image_mask": {
                "left_base_0_rgb": np.True_,
                "right_base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            }
        }

        if "frame_index" in data:
            inputs["frame_index"] = data["frame_index"]

        if "episode_index" in data:
            inputs["episode_index"] = data["episode_index"]

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class YAMOutputs(transforms.DataTransformFn):
    """Converts model outputs back to YAM action format (14D)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :14])}
