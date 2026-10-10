import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class TianjiJointInputs(transforms.DataTransformFn):
    """Inputs for Tianji dual-arm joint-control data.

    Expected after repack:
    - images: head, hik_camera_6, hik_camera_7
    - state: [22] = joints[14] + eef_left.xyz[3] + eef_right.xyz[3] + gripper L/R[2]
    - actions: [horizon, 16] = joints[14] + gripper L/R[2]
    """

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("head", "hik_camera_6", "hik_camera_7")

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": np.asarray(data["state"], dtype=np.float32),
        }

        if "images" in data:
            in_images = data["images"]
            if set(in_images) - set(self.EXPECTED_CAMERAS):
                raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

            images = {name: _to_hwc_uint8(image) for name, image in in_images.items()}
            base_image = images["head"]

            out_images = {
                "base_0_rgb": base_image,
            }
            image_masks = {
                "base_0_rgb": np.True_,
            }

            extra_images = {
                "left_wrist_0_rgb": "hik_camera_6",
                "right_wrist_0_rgb": "hik_camera_7",
            }
            for dest, source in extra_images.items():
                if source in images:
                    out_images[dest] = images[source]
                    image_masks[dest] = np.True_
                else:
                    out_images[dest] = np.zeros_like(base_image)
                    image_masks[dest] = np.False_

            inputs["image"] = out_images
            inputs["image_mask"] = image_masks

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "cas_sigma_target" in data:
            inputs["cas_sigma_target"] = np.asarray(data["cas_sigma_target"], dtype=np.float32)

        if "pps_weight" in data:
            inputs["pps_weight"] = np.asarray(data["pps_weight"], dtype=np.float32)

        return inputs


@dataclasses.dataclass(frozen=True)
class TianjiJointOutputs(transforms.DataTransformFn):
    action_dim: int = 16

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim], dtype=np.float32)}


@dataclasses.dataclass(frozen=True)
class TianjiEefPoseInputs(TianjiJointInputs):
    """Inputs for Tianji dual-arm EEF-pose control data.

    Expected after repack:
    - images: head, hik_camera_6, hik_camera_7
    - state: [16] = left pose[7] + left gripper[1] + right pose[7] + right gripper[1]
    - actions: [horizon, 16] = left target pose[7] + left gripper cmd[1]
      + right target pose[7] + right gripper cmd[1]
    """


@dataclasses.dataclass(frozen=True)
class TianjiEefPoseOutputs(TianjiJointOutputs):
    """Outputs for Tianji dual-arm EEF-pose control data."""


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
