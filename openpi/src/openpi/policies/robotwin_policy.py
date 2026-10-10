"""RoboTwin 2.0 transforms for the native ALOHA joint-action space."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class RobotwinInputs(transforms.DataTransformFn):
    """Convert the legacy PACE dataset layout to RoboTwin's native 14-D layout.

    The legacy dataset stores state as ``[L6, R6, pad2, Leef3, Reef3, Lg, Rg]``
    and action as ``[L6, R6, pad2, Lg, Rg]``.  The native ALOHA layout used by
    the pretrained pi0.5 model and RoboTwin is ``[L6, Lg, R6, Rg]``.

    Legacy training videos also need a red/blue swap because they were decoded
    with PIL from JPEGs encoded by RoboTwin's OpenCV recorder.  Live evaluation
    observations have no action field and are already normal RGB, so the swap
    is applied only to training examples.
    """

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("head", "hik_camera_6", "hik_camera_7")

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape[-1] != 22:
            raise ValueError(f"Expected legacy RoboTwin state dimension 22, got {state.shape}")
        native_state = np.concatenate(
            (state[..., :6], state[..., 20:21], state[..., 6:12], state[..., 21:22]),
            axis=-1,
        )

        inputs: dict = {"state": native_state}
        is_training = "actions" in data
        if "images" in data:
            in_images = data["images"]
            if set(in_images) - set(self.EXPECTED_CAMERAS):
                raise ValueError(
                    f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}"
                )
            images = {name: _to_hwc_uint8(image) for name, image in in_images.items()}
            if is_training:
                images = {
                    name: np.ascontiguousarray(image[..., ::-1]) for name, image in images.items()
                }
            inputs["image"] = {
                "base_0_rgb": images["head"],
                "left_wrist_0_rgb": images["hik_camera_6"],
                "right_wrist_0_rgb": images["hik_camera_7"],
            }
            inputs["image_mask"] = {name: np.True_ for name in inputs["image"]}

        if is_training:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != 16:
                raise ValueError(f"Expected legacy RoboTwin action dimension 16, got {actions.shape}")
            inputs["actions"] = np.concatenate(
                (
                    actions[..., :6],
                    actions[..., 14:15],
                    actions[..., 6:12],
                    actions[..., 15:16],
                ),
                axis=-1,
            )

        for key in ("prompt", "cas_sigma_target", "pps_weight"):
            if key in data:
                inputs[key] = data[key]
        return inputs


@dataclasses.dataclass(frozen=True)
class RobotwinOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :14], dtype=np.float32)}


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
