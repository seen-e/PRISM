import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class RLBenchInputs(transforms.DataTransformFn):
    """Map RLBench's three-camera, single-arm samples to model inputs."""

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("front", "wrist", "left_shoulder")

    def __call__(self, data: dict) -> dict:
        inputs = {"state": np.asarray(data["state"], dtype=np.float32)}

        if "images" in data:
            images = {name: _to_hwc_uint8(image) for name, image in data["images"].items()}
            if set(images) - set(self.EXPECTED_CAMERAS):
                raise ValueError(f"Expected cameras {self.EXPECTED_CAMERAS}, got {tuple(images)}")

            base_image = images["front"]
            inputs["image"] = {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": images.get("wrist", np.zeros_like(base_image)),
                "right_wrist_0_rgb": images.get("left_shoulder", np.zeros_like(base_image)),
            }
            inputs["image_mask"] = {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.bool_("wrist" in images),
                "right_wrist_0_rgb": np.bool_("left_shoulder" in images),
            }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "pps_weight" in data:
            inputs["pps_weight"] = np.asarray(data["pps_weight"], dtype=np.float32)
        return inputs


@dataclasses.dataclass(frozen=True)
class RLBenchOutputs(transforms.DataTransformFn):
    action_dim: int = 8

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
