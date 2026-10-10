"""Policy transforms for the Sculptor single-arm suction robot."""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class SculptorInputs(transforms.DataTransformFn):
    """Select the arm state and map the three Sculptor cameras to pi0.5."""

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam3", "cam9", "cam10")

    def __call__(self, data: dict) -> dict:
        raw_state = np.asarray(data["state"], dtype=np.float32)
        if raw_state.shape[-1] != 11:
            raise ValueError(f"Expected 11-D Sculptor state, got {raw_state.shape}")

        # Dataset layout: mobile-base[3], arm joints[6], mast[1], pressure[1].
        # The policy controls only the six arm joints and suction channel.
        inputs = {
            "state": np.concatenate((raw_state[..., 3:9], raw_state[..., 10:11]), axis=-1),
        }

        if "images" in data:
            images = {name: _to_hwc_uint8(image) for name, image in data["images"].items()}
            if set(images) - set(self.EXPECTED_CAMERAS):
                raise ValueError(f"Expected cameras {self.EXPECTED_CAMERAS}, got {tuple(images)}")

            base_image = images["cam3"]
            inputs["image"] = {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": images.get("cam9", np.zeros_like(base_image)),
                "right_wrist_0_rgb": images.get("cam10", np.zeros_like(base_image)),
            }
            inputs["image_mask"] = {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.bool_("cam9" in images),
                "right_wrist_0_rgb": np.bool_("cam10" in images),
            }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "pps_weight" in data:
            inputs["pps_weight"] = np.asarray(data["pps_weight"], dtype=np.float32)
        return inputs


@dataclasses.dataclass(frozen=True)
class SculptorOutputs(transforms.DataTransformFn):
    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim], dtype=np.float32)}


def _to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).clip(0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = einops.rearrange(image, "c h w -> h w c")
    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return image
