"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import os
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.rlbench_policy as rlbench_policy
import openpi.policies.robotwin_policy as robotwin_policy
import openpi.policies.sculptor_policy as sculptor_policy
import openpi.policies.tianji_policy as tianji_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)
    # Frame stride between consecutive actions in a LeRobot action chunk.
    action_stride: int = 1

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # PACE / CAS: path to directory containing phase_sigma.json and
    # phase_boundaries.json produced by openpi.pace.compute_phase_sigma.
    # When set, each training sample gets a `cas_sigma_target` field.
    cas_asset_dir: str | None = None
    # PACE / PPS: path to pps_weights.npy produced by compute_pps_weights.
    # When set, each training sample gets a `pps_weight` scalar field.
    pps_weight_path: str | None = None
    # Optional episode-level holdout. Frames from these LeRobot episode indices
    # are excluded from training while retaining their original global indices
    # for PPS lookup and offline evaluation.
    exclude_episode_indices: tuple[int, ...] = ()

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Optional LeRobot video decoder override. Useful when torchcodec is installed but missing FFmpeg runtime libs.
    video_backend: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None
    # Frame stride between consecutive actions in a LeRobot action chunk.
    action_stride: int = 1
    # Optional frame-aligned PRISM targets. Keeping this on the base factory
    # makes PPS available to every LeRobot data adapter.
    pps_weight_path: str | None = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if self.action_stride < 1:
            raise ValueError(f"action_stride must be positive, got {self.action_stride}")
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
            action_stride=self.action_stride,
            pps_weight_path=self.pps_weight_path,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False
    pps_weight_path: str | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                        "pps_weight": "pps_weight",
                    },
                    optional_keys=frozenset({"actions", "pps_weight"}),
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            pps_weight_path=self.pps_weight_path,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotTianjiJointDataConfig(DataConfigFactory):
    # Tianji action is [14 dual-arm joints, gripper_left, gripper_right].
    action_dim: int = 16
    use_delta_joint_actions: bool = True

    action_sequence_keys: Sequence[str] = ("action",)

    # PACE / CAS: path to phase_sigma.json + phase_boundaries.json directory.
    cas_asset_dir: str | None = None
    # PACE / PPS: path to pps_weights.npy.
    pps_weight_path: str | None = None
    # Episode indices reserved for offline evaluation.
    exclude_episode_indices: tuple[int, ...] = ()

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "head": "observation.images.head",
                            "hik_camera_6": "observation.images.hik_camera_6",
                            "hik_camera_7": "observation.images.hik_camera_7",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                        # PACE fields injected by _CachedLeRobotV3Dataset; pass through repack.
                        "cas_sigma_target": "cas_sigma_target",
                        "pps_weight": "pps_weight",
                    },
                    optional_keys=frozenset({"actions", "cas_sigma_target", "pps_weight"}),
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[tianji_policy.TianjiJointInputs()],
            outputs=[tianji_policy.TianjiJointOutputs(action_dim=self.action_dim)],
        )

        if self.use_delta_joint_actions:
            # Only joint targets become deltas. The final two gripper command dims stay absolute.
            delta_action_mask = _transforms.make_bool_mask(14, -2)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            cas_asset_dir=self.cas_asset_dir,
            pps_weight_path=self.pps_weight_path,
            exclude_episode_indices=self.exclude_episode_indices,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotRobotwinDataConfig(DataConfigFactory):
    """RoboTwin data in the native ALOHA 14-D state-action convention."""

    action_sequence_keys: Sequence[str] = ("action",)
    pps_weight_path: str | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "head": "observation.images.head",
                            "hik_camera_6": "observation.images.hik_camera_6",
                            "hik_camera_7": "observation.images.hik_camera_7",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                        "pps_weight": "pps_weight",
                    },
                    optional_keys=frozenset({"actions", "pps_weight"}),
                )
            ]
        )
        delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
        data_transforms = _transforms.Group(
            inputs=[
                robotwin_policy.RobotwinInputs(),
                _transforms.DeltaActions(delta_action_mask),
            ],
            outputs=[
                _transforms.AbsoluteActions(delta_action_mask),
                robotwin_policy.RobotwinOutputs(),
            ],
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=self.action_sequence_keys,
            pps_weight_path=self.pps_weight_path,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotTianjiEefPoseDataConfig(DataConfigFactory):
    # Tianji EEF pose action is [left pose7, left gripper, right pose7, right gripper].
    action_dim: int = 16
    use_chunkwise_delta_pose_actions: bool = True

    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "head": "observation.images.head",
                            "hik_camera_6": "observation.images.hik_camera_6",
                            "hik_camera_7": "observation.images.hik_camera_7",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                        # PACE fields injected by _CachedLeRobotV3Dataset; pass through repack.
                        "cas_sigma_target": "cas_sigma_target",
                        "pps_weight": "pps_weight",
                    },
                    optional_keys=frozenset({"actions", "cas_sigma_target", "pps_weight"}),
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[tianji_policy.TianjiEefPoseInputs()],
            outputs=[tianji_policy.TianjiEefPoseOutputs(action_dim=self.action_dim)],
        )

        if self.use_chunkwise_delta_pose_actions:
            # Chunk-wise delta: every horizon row is relative to state_t.
            # Gripper command dims stay absolute.
            delta_action_mask = _transforms.make_bool_mask(7, -1, 7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotRLBenchDataConfig(DataConfigFactory):
    """RLBench single-arm joint control with PACE precision labels."""

    action_dim: int = 8
    use_delta_joint_actions: bool = True
    action_sequence_keys: Sequence[str] = ("action",)
    pps_weight_path: str | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "front": "observation.images.front",
                            "wrist": "observation.images.wrist",
                            "left_shoulder": "observation.images.left_shoulder",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                        "pps_weight": "pps_weight",
                    },
                    optional_keys=frozenset({"actions", "pps_weight"}),
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[rlbench_policy.RLBenchInputs()],
            outputs=[rlbench_policy.RLBenchOutputs(action_dim=self.action_dim)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=self.action_sequence_keys,
            pps_weight_path=self.pps_weight_path,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotSculptorDataConfig(DataConfigFactory):
    """Sculptor single-arm joint control with suction and PRISM labels."""

    action_dim: int = 7
    use_delta_joint_actions: bool = True
    action_sequence_keys: Sequence[str] = ("action",)
    pps_weight_path: str | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam3": "observation.images.cam3",
                            "cam9": "observation.images.cam9",
                            "cam10": "observation.images.cam10",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                        "pps_weight": "pps_weight",
                    },
                    optional_keys=frozenset({"actions", "pps_weight"}),
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[sculptor_policy.SculptorInputs()],
            outputs=[sculptor_policy.SculptorOutputs(action_dim=self.action_dim)],
        )
        if self.use_delta_joint_actions:
            # Six joint targets are relative to current joints; suction is absolute.
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=self.action_sequence_keys,
            pps_weight_path=self.pps_weight_path,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, save checkpoints only at these steps (overrides save_interval).
    save_on_steps: tuple[int, ...] | None = None
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000
    # Override Orbax's rolling checkpoint count. Preserved keep_period checkpoints
    # are retained in addition to this rolling count.
    checkpoint_max_to_keep: int | None = None

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # PACE / CAS: if true, run CAS warm-up (Phase 2) instead of standard
    # flow-matching training. Only scale_head receives gradients; the rest
    # of the backbone should be frozen via get_cas_warmup_freeze_filter().
    cas_warmup: bool = False

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1
    # Controls how batches and activation sharding constraints use the mesh.
    # "all" preserves the original behavior: PartitionSpec(("batch", "fsdp")).
    # "batch_only" shards only on the batch axis, useful for isolating FSDP collective/layout issues.
    data_axis_mode: Literal["all", "batch_only"] = "all"
    # Minimum tensor size considered for automatic FSDP parameter/optimizer sharding.
    fsdp_min_size_mbytes: int = 4
    # Debug knob: keep embedding-like arrays replicated while still sharding other large arrays.
    replicate_embeddings: bool = False

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


def _robotwin50_generalist_pace_config() -> TrainConfig:
    """Build the unified 50-task config used by the RoboTwin 2.0 protocol."""
    repo_id = "local/robotwin50_clean"
    return TrainConfig(
        name="pi05_robotwin50_pace_v5",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            loss_action_dim=16,
            max_token_len=200,
            use_cas=False,
            use_da_fta=False,
            use_pps=False,
            use_psw=True,
            psw_gamma=2.0,
            psw_tau_0=1.0,
            psw_p=2.0,
            use_cpr=True,
            cpr_lambda=0.1,
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id=repo_id,
            # RoboTwin is already converted at the 15 Hz control rate.
            action_stride=1,
            pps_weight_path="assets/pace/pps/robotwin50/pps_weights_universal.npy",
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=15_000,
        batch_size=192,
        num_workers=0,
        log_interval=10,
        save_on_steps=tuple(range(1_000, 15_001, 1_000)),
        keep_period=8_000,
        checkpoint_max_to_keep=1,
        fsdp_devices=1,
        data_axis_mode="batch_only",
    )


def _robotwin50_generalist_pace_native_config() -> TrainConfig:
    """Corrected RoboTwin 2.0 config with native 14-D actions and a 15-step horizon."""
    return TrainConfig(
        name="pi05_robotwin50_pace_native",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            loss_action_dim=14,
            max_token_len=200,
            use_cas=False,
            use_da_fta=False,
            use_pps=False,
            use_psw=True,
            psw_gamma=2.0,
            psw_tau_0=1.0,
            psw_p=2.0,
            use_cpr=True,
            cpr_lambda=0.1,
        ),
        data=LeRobotRobotwinDataConfig(
            repo_id="local/robotwin50_clean_v2",
            action_stride=1,
            pps_weight_path="assets/pace/pps/robotwin50/pps_weights_universal.npy",
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=15_000,
        batch_size=192,
        num_workers=4,
        log_interval=10,
        save_on_steps=tuple(range(1_000, 15_001, 1_000)),
        keep_period=8_000,
        checkpoint_max_to_keep=1,
        fsdp_devices=1,
        data_axis_mode="batch_only",
    )


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    TrainConfig(
        # Public, environment-driven PRISM recipe. This is the same current
        # OpenPI/PRISM training stack as the task-specific internal recipes,
        # without embedding private dataset paths.
        name="pi05_prism",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            loss_action_dim=int(os.environ.get("PRISM_LOSS_ACTION_DIM", "14")),
            use_cas=False,
            use_da_fta=False,
            use_pps=False,
            use_psw=True,
            psw_gamma=float(os.environ.get("PRISM_PSW_GAMMA", "2.0")),
            psw_tau_0=float(os.environ.get("PRISM_PSW_TAU_0", "1.0")),
            psw_p=float(os.environ.get("PRISM_PSW_P", "2.0")),
            psw_use_snr=True,
            use_cpr=True,
            cpr_lambda=float(os.environ.get("PRISM_CPR_LAMBDA", "0.1")),
        ),
        data=LeRobotAlohaDataConfig(
            repo_id=os.environ.get("PRISM_REPO_ID", "set/PRISM_REPO_ID"),
            pps_weight_path=os.environ.get("PRISM_PPS_PATH", "pps_weights.npy"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.environ.get("PRISM_BASE_CHECKPOINT", "gs://openpi-assets/checkpoints/pi05_base/params")
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=int(os.environ.get("PRISM_NUM_TRAIN_STEPS", "15000")),
        batch_size=int(os.environ.get("PRISM_BATCH_SIZE", "192")),
        num_workers=int(os.environ.get("PRISM_NUM_WORKERS", "0")),
        log_interval=10,
        save_interval=1000,
        keep_period=8000,
        assets_base_dir=os.environ.get("PRISM_ASSETS_DIR", "./assets"),
        checkpoint_base_dir=os.environ.get("PRISM_CHECKPOINT_DIR", "./checkpoints"),
    ),
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_libero_prism",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            use_psw=True,
            psw_gamma=2.0,
            psw_tau_0=1.0,
            psw_p=2.0,
            use_cpr=True,
            cpr_lambda=0.1,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            assets=AssetsConfig(assets_dir="./assets/pi05_libero"),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            pps_weight_path="assets/pace/pps/libero/pps_weights_universal.npy",
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=20_000,
        save_on_steps=(10_000, 20_000),
    ),
    TrainConfig(
        name="pi05_tianji_joint_sft",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            loss_action_dim=16,
            max_token_len=200,
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/tianji_joint_eef_20260523",
            assets=AssetsConfig(assets_dir="./assets/pi05_tianji_joint_sft"),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=20_000,
        batch_size=32,
        num_workers=4,
        log_interval=50,
        save_interval=1000,
        keep_period=5000,
    ),
    TrainConfig(
        name="pi05_tianji_joint_sft_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            loss_action_dim=16,
            max_token_len=200,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/tianji_joint_eef_20260523",
            assets=AssetsConfig(assets_dir="./assets/pi05_tianji_joint_sft"),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            loss_action_dim=16,
            max_token_len=200,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=20_000,
        batch_size=8,
        num_workers=4,
        log_interval=50,
        save_interval=1000,
        keep_period=5000,
    ),
    TrainConfig(
        name="pi05_tianji_joint_20260604_sft",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            loss_action_dim=16,
            max_token_len=200,
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/tianji_joint_eef_20260604",
            assets=AssetsConfig(assets_dir="./assets/pi05_tianji_joint_20260604_sft"),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=20_000,
        batch_size=32,
        num_workers=4,
        log_interval=50,
        save_interval=1000,
        keep_period=5000,
        fsdp_devices=4,
    ),
    TrainConfig(
        name="pi05_tianji_joint_537_20260610_sft",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            loss_action_dim=16,
            max_token_len=200,
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/tianji_joint_537_20260610",
            assets=AssetsConfig(assets_dir="./assets/pi05_tianji_joint_537_20260610_sft"),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=30_000,
        batch_size=176,
        num_workers=4,
        log_interval=10,
        save_interval=1000,
        keep_period=5000,
        fsdp_devices=4,
    ),
    TrainConfig(
        name="pi05_tianji_joint_500_20260612_sft",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            loss_action_dim=16,
            max_token_len=200,
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/tianji_joint_504_newop_20260612",
            assets=AssetsConfig(assets_dir="./assets/pi05_tianji_joint_500_20260612_sft"),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=50_000,
        batch_size=176,
        num_workers=4,
        log_interval=10,
        save_interval=1000,
        keep_period=5000,
        fsdp_devices=4,
    ),
    TrainConfig(
        name="pi05_tianji_joint_500_20260612_stride2_sft",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            loss_action_dim=16,
            max_token_len=200,
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/tianji_joint_504_newop_20260612",
            action_stride=2,
            assets=AssetsConfig(assets_dir="./assets/pi05_tianji_joint_500_20260612_stride2_sft"),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=50_000,
        batch_size=176,
        num_workers=4,
        log_interval=10,
        save_interval=1000,
        keep_period=5000,
        fsdp_devices=4,
    ),
    TrainConfig(
        name="pi05_tianji_joint_500_20260612_fps15_chunk15_sft",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            loss_action_dim=16,
            max_token_len=200,
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/tianji_joint_504_newop_20260612",
            action_stride=2,
            assets=AssetsConfig(assets_dir="./assets/pi05_tianji_joint_500_20260612_fps15_chunk15_sft"),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=50_000,
        batch_size=176,
        num_workers=4,
        log_interval=10,
        save_interval=1000,
        keep_period=5000,
        fsdp_devices=4,
    ),
    TrainConfig(
        name="pi05_tianji_joint_0629_438eps_dagge_human_expert_sft",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            loss_action_dim=16,
            max_token_len=200,
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/0629_438eps_dagge_human_expert",
            assets=AssetsConfig(assets_dir="./assets/pi05_tianji_joint_0629_438eps_dagge_human_expert_sft"),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=50_000,
        batch_size=176,
        num_workers=4,
        log_interval=10,
        save_interval=1000,
        keep_period=5000,
        fsdp_devices=4,
    ),
    TrainConfig(
        name="pi05_tianji_joint_0629_438eps_dagge_human_expert_stride2_sft",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            loss_action_dim=16,
            max_token_len=200,
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/0629_438eps_dagge_human_expert",
            action_stride=2,
            assets=AssetsConfig(
                assets_dir="./assets/pi05_tianji_joint_0629_438eps_dagge_human_expert_stride2_sft"
            ),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=50_000,
        batch_size=512,
        # Host RAM on this node is very limited (10GiB). Each dataloader worker process
        # costs ~800MB just importing jax/torch, so keep this small; the preprocessed
        # JPEG image cache (see scripts/preprocess_video_cache.py) makes per-sample reads
        # fast enough that we don't need many workers to keep the GPUs fed.
        num_workers=0,
        log_interval=10,
        save_interval=1000,
        keep_period=5000,
        fsdp_devices=1,
        # With fsdp_devices=1 the mesh is (batch=8, fsdp=1). PartitionSpec(("batch", "fsdp"))
        # breaks SigLIP nn.scan XLA compile on JAX 0.5.3 (u32[512,2] sharding mismatch).
        data_axis_mode="batch_only",
    ),
    # ---- PACE CAS warm-up (Phase 2) ----------------------------------------
    TrainConfig(
        name="pi05_tianji_cas_warmup",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            loss_action_dim=16,
            max_token_len=200,
            use_cas=True,
            use_da_fta=False,  # warm-up only trains scale_head via compute_cas_warmup_loss
            # cas_init_sigma is populated at launch time from phase_sigma.json.
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/0629_438eps_dagge_human_expert",
            action_stride=2,
            cas_asset_dir="assets/pace/cas/0629_438eps",
            assets=AssetsConfig(
                assets_dir="./assets/pi05_tianji_joint_0629_438eps_dagge_human_expert_stride2_sft",
            ),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        # Load from the SFT checkpoint produced by the earlier pi05 training run.
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_tianji_joint_0629_438eps_dagge_human_expert_stride2_sft/b192_f1_h15_0629/5000/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=15,
            loss_action_dim=16, max_token_len=200, use_cas=True,
        ).get_cas_warmup_freeze_filter(),
        cas_warmup=True,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=100,
            peak_lr=1e-3,
            decay_steps=5_000,
            decay_lr=1e-4,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=3_000,
        batch_size=192,
        num_workers=0,
        log_interval=10,
        save_on_steps=(1_000, 3_000),
        fsdp_devices=1,
        data_axis_mode="batch_only",
    ),
    # ---- PACE DA-FTA end-to-end (CAS + DA-FTA, no warmup) --------------------
    # ScaleHead bias-initialised to global action std; trained jointly with the
    # backbone via DA-FTA loss gradients.  No phase prior, no warmup stage.
    TrainConfig(
        name="pi05_tianji_da_fta_e2e",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            loss_action_dim=16,
            max_token_len=200,
            use_cas=True,
            use_da_fta=True,
            da_fta_sigma_mode="cas",
            cas_init_sigma=(
                0.18539337898682362, 0.21845398819715747, 0.11086866619407716, 0.20364291520733882,
                0.2247156007602922, 0.2431097809096788, 0.48309024536263534, 0.07706123407019662,
                0.15272096818633282, 0.11712216117916195, 0.0963498634092118, 0.07736279768235466,
                0.14435638088184768, 0.26837224316114083, 0.28773935205929574, 0.4504381646454872,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ),
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/0629_438eps_dagge_human_expert",
            action_stride=2,
            assets=AssetsConfig(
                assets_dir="./assets/pi05_tianji_joint_0629_438eps_dagge_human_expert_stride2_sft",
            ),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=20_000,
        batch_size=192,
        num_workers=0,
        log_interval=10,
        save_on_steps=(10_000, 20_000),
        keep_period=None,
        fsdp_devices=1,
        data_axis_mode="batch_only",
    ),
    # ---- PACE DA-FTA end-to-end + PPS (CAS + DA-FTA + sample weighting) ------
    # Same as pi05_tianji_da_fta_e2e with Phase-4 PPS weights (slow motion upweighted).
    TrainConfig(
        name="pi05_tianji_da_fta_e2e_pps",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            loss_action_dim=16,
            max_token_len=200,
            use_cas=True,
            use_da_fta=True,
            da_fta_sigma_mode="cas",
            use_pps=True,
            cas_init_sigma=(
                0.18539337898682362, 0.21845398819715747, 0.11086866619407716, 0.20364291520733882,
                0.2247156007602922, 0.2431097809096788, 0.48309024536263534, 0.07706123407019662,
                0.15272096818633282, 0.11712216117916195, 0.0963498634092118, 0.07736279768235466,
                0.14435638088184768, 0.26837224316114083, 0.28773935205929574, 0.4504381646454872,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ),
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/0629_438eps_dagge_human_expert",
            action_stride=2,
            pps_weight_path="assets/pace/pps/0629_438eps/pps_weights.npy",
            assets=AssetsConfig(
                assets_dir="./assets/pi05_tianji_joint_0629_438eps_dagge_human_expert_stride2_sft",
            ),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=20_000,
        batch_size=192,
        num_workers=0,
        log_interval=10,
        save_on_steps=(10_000, 20_000),
        keep_period=None,
        fsdp_devices=1,
        data_axis_mode="batch_only",
    ),
    # ---- PACE DA-FTA-Safe (Phase 3, deprecated warmup path) -----------------
    # Kept for reference.  Prefer pi05_tianji_da_fta_e2e for new experiments.
    TrainConfig(
        name="pi05_tianji_da_fta_safe",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            loss_action_dim=16,
            max_token_len=200,
            use_cas=True,
            use_da_fta=True,
            da_fta_sigma_mode="cas",
            cas_init_sigma=(
                0.18539337898682362, 0.21845398819715747, 0.11086866619407716, 0.20364291520733882,
                0.2247156007602922, 0.2431097809096788, 0.48309024536263534, 0.07706123407019662,
                0.15272096818633282, 0.11712216117916195, 0.0963498634092118, 0.07736279768235466,
                0.14435638088184768, 0.26837224316114083, 0.28773935205929574, 0.4504381646454872,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ),
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/0629_438eps_dagge_human_expert",
            action_stride=2,
            assets=AssetsConfig(
                assets_dir="./assets/pi05_tianji_joint_0629_438eps_dagge_human_expert_stride2_sft",
            ),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_tianji_joint_0629_438eps_dagge_human_expert_stride2_sft/b192_f1_h15_0629/5000/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=10_000,
        batch_size=192,
        num_workers=0,
        log_interval=10,
        save_interval=1000,
        keep_period=5000,
        fsdp_devices=1,
        data_axis_mode="batch_only",
    ),
    # ---- DA-FTA with global fixed sigma (no CAS head) -----------------------
    TrainConfig(
        name="pi05_tianji_da_fta_global",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            loss_action_dim=16,
            max_token_len=200,
            use_cas=False,
            use_da_fta=True,
            da_fta_sigma_mode="global",
            # Placeholder; override at launch from phase_sigma.json global_sigma_init.
            da_fta_global_sigma=(
                0.18539337898682362, 0.21845398819715747, 0.11086866619407716, 0.20364291520733882,
                0.2247156007602922, 0.2431097809096788, 0.48309024536263534, 0.07706123407019662,
                0.15272096818633282, 0.11712216117916195, 0.0963498634092118, 0.07736279768235466,
                0.14435638088184768, 0.26837224316114083, 0.28773935205929574, 0.4504381646454872,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ),
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/0629_438eps_dagge_human_expert",
            action_stride=2,
            assets=AssetsConfig(
                assets_dir="./assets/pi05_tianji_joint_0629_438eps_dagge_human_expert_stride2_sft",
            ),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_tianji_joint_0629_438eps_dagge_human_expert_stride2_sft/b192_f1_h15_0629/5000/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=10_000,
        batch_size=192,
        num_workers=0,
        log_interval=10,
        save_interval=1000,
        keep_period=5000,
        fsdp_devices=1,
        data_axis_mode="batch_only",
    ),
    # ---- DA-FTA with offline phase sigma (no CAS head) ----------------------
    TrainConfig(
        name="pi05_tianji_da_fta_phase_target",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            loss_action_dim=16,
            max_token_len=200,
            use_cas=False,
            use_da_fta=True,
            da_fta_sigma_mode="phase_target",
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/0629_438eps_dagge_human_expert",
            action_stride=2,
            cas_asset_dir="assets/pace/cas/0629_438eps",
            assets=AssetsConfig(
                assets_dir="./assets/pi05_tianji_joint_0629_438eps_dagge_human_expert_stride2_sft",
            ),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_tianji_joint_0629_438eps_dagge_human_expert_stride2_sft/b192_f1_h15_0629/5000/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=10_000,
        batch_size=192,
        num_workers=0,
        log_interval=10,
        save_interval=1000,
        keep_period=5000,
        fsdp_devices=1,
        data_axis_mode="batch_only",
    ),
    # ---- PACE v5 (PSW + CPR, universal PPS) ---------------------------------
    # Standard FM + precision-scaled weighting + CPR regularization.
    # No legacy CAS / DA-FTA / flat PPS.  Inference identical to baseline pi0.5.
    # PPS offline rho loaded via data.pps_weight_path -> batch field pps_weight.
    # use_psw applies v5 PSW (w = 1 + gamma * rho * Psi); legacy use_pps is flat (1+alpha*rho).
    TrainConfig(
        name="pi05_tianji_pace_v5",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            loss_action_dim=16,
            max_token_len=200,
            use_cas=False,
            use_da_fta=False,
            use_pps=False,
            use_psw=True,
            psw_gamma=1.0,
            psw_tau_0=1.0,
            psw_p=2.0,
            use_cpr=True,
            cpr_lambda=0.1,
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/1210",
            action_stride=2,
            pps_weight_path="assets/pace/pps/111_1099_merged/pps_weights_universal.npy",
            assets=AssetsConfig(
                assets_dir="./assets/pi05_tianji_1210_stride2_sft",
            ),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=15_000,
        batch_size=192,
        num_workers=0,
        log_interval=10,
        save_on_steps=(8_000, 15_000),
        keep_period=None,
        fsdp_devices=1,
        data_axis_mode="batch_only",
    ),
    # ---- PACE v5, 438 human/expert dataset (8-GPU training) ----------------
    # Keep this separate from the 1210 configuration above: its normalization
    # assets and universal PPS labels are dataset-specific.
    TrainConfig(
        name="pi05_tianji_pace_v5_438",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            loss_action_dim=16,
            max_token_len=200,
            use_cas=False,
            use_da_fta=False,
            use_pps=False,
            use_psw=True,
            psw_gamma=2.0,
            psw_tau_0=1.0,
            psw_p=2.0,
            use_cpr=True,
            cpr_lambda=0.1,
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/0629_438eps_dagge_human_expert",
            action_stride=2,
            pps_weight_path="assets/pace/pps/0629_438eps/pps_weights_universal.npy",
            assets=AssetsConfig(
                assets_dir="./assets/pi05_tianji_joint_0629_438eps_dagge_human_expert_stride2_sft",
            ),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=15_000,
        batch_size=192,
        num_workers=0,
        log_interval=10,
        save_on_steps=(8_000, 15_000),
        keep_period=None,
        fsdp_devices=1,
        data_axis_mode="batch_only",
    ),
    # ---- PACE v5, cleaned cable-tie dataset (8-GPU training) ---------------
    TrainConfig(
        name="pi05_tianji_pace_v5_462_clean",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            loss_action_dim=16,
            max_token_len=200,
            use_cas=False,
            use_da_fta=False,
            use_pps=False,
            use_psw=True,
            psw_gamma=2.0,
            psw_tau_0=1.0,
            psw_p=2.0,
            use_cpr=True,
            cpr_lambda=0.1,
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/tianji_cabletie_462eps_gop2_clean",
            action_stride=2,
            pps_weight_path="assets/pace/pps/tianji_cabletie_462_clean/pps_weights_universal.npy",
            assets=AssetsConfig(
                assets_dir="./assets/pi05_tianji_cabletie_462_clean_stride2_sft",
            ),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=15_000,
        batch_size=192,
        num_workers=0,
        log_interval=10,
        save_on_steps=tuple(range(1_000, 15_001, 1_000)),
        keep_period=8_000,
        checkpoint_max_to_keep=1,
        fsdp_devices=1,
        data_axis_mode="batch_only",
    ),
    # ---- PRISM, Sculptor single-arm suction dataset (8-GPU training) ------
    TrainConfig(
        name="pi05_sculptor_prism",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            loss_action_dim=7,
            max_token_len=200,
            use_cas=False,
            use_da_fta=False,
            use_pps=False,
            use_psw=True,
            psw_gamma=2.0,
            psw_tau_0=1.0,
            psw_p=2.0,
            use_cpr=True,
            cpr_lambda=0.1,
        ),
        data=LeRobotSculptorDataConfig(
            repo_id="local/sculptor_0903_60eps",
            action_stride=2,
            pps_weight_path="assets/pace/pps/sculptor_0903_60eps/pps_weights_universal.npy",
            assets=AssetsConfig(
                assets_dir="./assets/pi05_sculptor_0903_60eps_stride2_sft",
            ),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=15_000,
        batch_size=192,
        num_workers=0,
        log_interval=10,
        save_on_steps=tuple(range(1_000, 15_001, 1_000)),
        keep_period=8_000,
        checkpoint_max_to_keep=1,
        fsdp_devices=1,
        data_axis_mode="batch_only",
    ),
    # ---- PACE v5, RLBench precision suite (8-GPU training) -----------------
    # Mirrors the current 438 PACE settings; only robot-specific observation
    # and action dimensions differ.
    TrainConfig(
        name="pi05_rlbench_pace_v5",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            loss_action_dim=8,
            max_token_len=200,
            use_cas=False,
            use_da_fta=False,
            use_pps=False,
            use_psw=True,
            psw_gamma=2.0,
            psw_tau_0=1.0,
            psw_p=2.0,
            use_cpr=True,
            cpr_lambda=0.1,
        ),
        data=LeRobotRLBenchDataConfig(
            repo_id="local/rlbench_pace_precision",
            action_stride=2,
            pps_weight_path="assets/pace/pps/rlbench_precision/pps_weights_universal.npy",
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=15_000,
        batch_size=192,
        num_workers=0,
        log_interval=10,
        save_on_steps=(8_000, 15_000),
        keep_period=None,
        fsdp_devices=1,
        data_axis_mode="batch_only",
    ),
    # ---- PACE v5, published RLBench 10-task multi-task protocol ------------
    TrainConfig(
        name="pi05_rlbench10_pace_v5",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            loss_action_dim=8,
            max_token_len=200,
            use_cas=False,
            use_da_fta=False,
            use_pps=False,
            use_psw=True,
            psw_gamma=2.0,
            psw_tau_0=1.0,
            psw_p=2.0,
            use_cpr=True,
            cpr_lambda=0.1,
        ),
        data=LeRobotRLBenchDataConfig(
            repo_id="local/rlbench10_pace",
            # Demonstrations and simulator rollout share one per-step clock.
            action_stride=1,
            pps_weight_path="assets/pace/pps/rlbench10/pps_weights_universal.npy",
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=15_000,
        batch_size=192,
        num_workers=0,
        log_interval=10,
        save_on_steps=(8_000, 15_000),
        keep_period=None,
        fsdp_devices=1,
        data_axis_mode="batch_only",
    ),
    # ---- PACE v5, RoboTwin dual-arm precision suite (8-GPU training) -------
    # RoboTwin is converted to the same joint / EEF / gripper state layout as
    # Tianji so Universal PPS uses its native dual-arm signals unchanged.
    TrainConfig(
        name="pi05_robotwin_pace_v5",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            loss_action_dim=16,
            max_token_len=200,
            use_cas=False,
            use_da_fta=False,
            use_pps=False,
            use_psw=True,
            psw_gamma=2.0,
            psw_tau_0=1.0,
            psw_p=2.0,
            use_cpr=True,
            cpr_lambda=0.1,
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/robotwin_pace_precision",
            action_stride=2,
            pps_weight_path="assets/pace/pps/robotwin_precision/pps_weights_universal.npy",
            assets=AssetsConfig(
                assets_dir="./assets/pi05_robotwin_pace_v5",
            ),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=15_000,
        batch_size=192,
        num_workers=0,
        log_interval=10,
        save_on_steps=(8_000, 15_000),
        keep_period=None,
        fsdp_devices=1,
        data_axis_mode="batch_only",
    ),
    # Qwen-RobotManip's RoboTwin protocol merges 50 clean demonstrations from
    # each of the 50 tasks and evaluates one generalist model on Easy and Hard.
    _robotwin50_generalist_pace_config(),
    _robotwin50_generalist_pace_native_config(),
    # ---- PACE v5 gamma=2 (stronger PSW on fine/slow frames) ------------------
    # Same as pi05_tianji_pace_v5 but psw_gamma=2.0 -> w in [1, 3] vs [1, 2].
    TrainConfig(
        name="pi05_tianji_pace_v5_g2",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            loss_action_dim=16,
            max_token_len=200,
            use_cas=False,
            use_da_fta=False,
            use_pps=False,
            use_psw=True,
            psw_gamma=2.0,
            psw_tau_0=1.0,
            psw_p=2.0,
            use_cpr=True,
            cpr_lambda=0.1,
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/1210",
            action_stride=2,
            pps_weight_path="assets/pace/pps/111_1099_merged/pps_weights_universal.npy",
            assets=AssetsConfig(
                assets_dir="./assets/pi05_tianji_1210_stride2_sft",
            ),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=15_000,
        batch_size=192,
        num_workers=0,
        log_interval=10,
        save_on_steps=(8_000, 15_000),
        keep_period=None,
        fsdp_devices=1,
        data_axis_mode="batch_only",
    ),
    # ---- PRISM real-robot insertion run with a fixed 10-episode holdout ----
    # The held-out complete episodes are reserved for post-training CPR/PPS
    # alignment and never sampled by the training loader.
    TrainConfig(
        name="pi05_tianji_prism_g2_cpr_holdout10",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            loss_action_dim=16,
            max_token_len=200,
            use_cas=False,
            use_da_fta=False,
            use_pps=False,
            use_psw=True,
            psw_gamma=2.0,
            psw_tau_0=1.0,
            psw_p=2.0,
            psw_use_snr=True,
            use_cpr=True,
            cpr_lambda=0.1,
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/1210",
            action_stride=2,
            pps_weight_path="assets/pace/pps/explore_1210/rho_speed_universal.npy",
            exclude_episode_indices=(31, 152, 210, 282, 461, 505, 577, 644, 878, 1152),
            assets=AssetsConfig(
                assets_dir="./assets/pi05_tianji_1210_stride2_holdout10_sft",
            ),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=15_000,
        batch_size=192,
        num_workers=0,
        log_interval=10,
        save_on_steps=tuple(range(1_000, 15_001, 1_000)),
        keep_period=8_000,
        checkpoint_max_to_keep=1,
        fsdp_devices=1,
        data_axis_mode="batch_only",
    ),
    # ---- Matched control: standard pi0.5 flow loss with CPR only -----------
    # This run shares the full PRISM data split and optimization schedule but
    # disables PSW, isolating the contribution of loss reweighting.
    TrainConfig(
        name="pi05_tianji_cpr_holdout10",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            loss_action_dim=16,
            max_token_len=200,
            use_cas=False,
            use_da_fta=False,
            use_pps=False,
            use_psw=False,
            use_cpr=True,
            cpr_lambda=0.1,
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/1210",
            action_stride=2,
            pps_weight_path="assets/pace/pps/explore_1210/rho_speed_universal.npy",
            exclude_episode_indices=(31, 152, 210, 282, 461, 505, 577, 644, 878, 1152),
            assets=AssetsConfig(
                assets_dir="./assets/pi05_tianji_1210_stride2_holdout10_sft",
            ),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=10_000,
        batch_size=192,
        num_workers=0,
        log_interval=10,
        save_on_steps=tuple(range(1_000, 10_001, 1_000)),
        keep_period=8_000,
        checkpoint_max_to_keep=1,
        fsdp_devices=1,
        data_axis_mode="batch_only",
    ),
    # ---- Matched standard pi0.5 baseline for the held-out insertion split --
    # This is the no-CPR, no-PSW reference used for component-wise analysis.
    TrainConfig(
        name="pi05_tianji_base_holdout10",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            loss_action_dim=16,
            max_token_len=200,
            use_cas=False,
            use_da_fta=False,
            use_pps=False,
            use_psw=False,
            use_cpr=False,
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/1210",
            action_stride=2,
            pps_weight_path="assets/pace/pps/explore_1210/rho_speed_universal.npy",
            exclude_episode_indices=(31, 152, 210, 282, 461, 505, 577, 644, 878, 1152),
            assets=AssetsConfig(
                assets_dir="./assets/pi05_tianji_1210_stride2_holdout10_sft",
            ),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=10_000,
        batch_size=192,
        num_workers=0,
        log_interval=10,
        save_on_steps=tuple(range(1_000, 10_001, 1_000)),
        keep_period=8_000,
        checkpoint_max_to_keep=1,
        fsdp_devices=1,
        data_axis_mode="batch_only",
    ),
    # ---- PRISM ablation: remove the SNR visibility gate --------------------
    # gamma is matched to the full model's expected PSW amplification on the
    # training split, so this intervention changes allocation rather than the
    # mean loss scale.
    TrainConfig(
        name="pi05_tianji_prism_g2_cpr_holdout10_no_snr",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            loss_action_dim=16,
            max_token_len=200,
            use_cas=False,
            use_da_fta=False,
            use_pps=False,
            use_psw=True,
            psw_gamma=1.412437254262,
            psw_tau_0=1.0,
            psw_p=2.0,
            psw_use_snr=False,
            use_cpr=True,
            cpr_lambda=0.1,
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/1210",
            action_stride=2,
            pps_weight_path="assets/pace/pps/explore_1210/rho_speed_universal.npy",
            exclude_episode_indices=(31, 152, 210, 282, 461, 505, 577, 644, 878, 1152),
            assets=AssetsConfig(
                assets_dir="./assets/pi05_tianji_1210_stride2_holdout10_sft",
            ),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=10_000,
        batch_size=192,
        num_workers=0,
        log_interval=10,
        save_on_steps=tuple(range(1_000, 10_001, 1_000)),
        keep_period=8_000,
        checkpoint_max_to_keep=1,
        fsdp_devices=1,
        data_axis_mode="batch_only",
    ),
    # ---- PRISM ablation: remove explicit clean-end flow-time emphasis -------
    # SNR remains sample dependent and is retained as the visibility gate.
    # gamma is mean-matched to the full PSW intervention on the training split.
    TrainConfig(
        name="pi05_tianji_prism_g2_cpr_holdout10_no_tau",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            loss_action_dim=16,
            max_token_len=200,
            use_cas=False,
            use_da_fta=False,
            use_pps=False,
            use_psw=True,
            psw_gamma=1.293986071901,
            psw_tau_0=1.0,
            psw_p=0.0,
            psw_use_snr=True,
            use_cpr=True,
            cpr_lambda=0.1,
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/1210",
            action_stride=2,
            pps_weight_path="assets/pace/pps/explore_1210/rho_speed_universal.npy",
            exclude_episode_indices=(31, 152, 210, 282, 461, 505, 577, 644, 878, 1152),
            assets=AssetsConfig(
                assets_dir="./assets/pi05_tianji_1210_stride2_holdout10_sft",
            ),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=10_000,
        batch_size=192,
        num_workers=0,
        log_interval=10,
        save_on_steps=tuple(range(1_000, 10_001, 1_000)),
        keep_period=8_000,
        checkpoint_max_to_keep=1,
        fsdp_devices=1,
        data_axis_mode="batch_only",
    ),
    # ---- PACE v5 gamma=3 (stronger PSW, full SNR Psi) ----------------------
    TrainConfig(
        name="pi05_tianji_pace_v5_g3",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            loss_action_dim=16,
            max_token_len=200,
            use_cas=False,
            use_da_fta=False,
            use_pps=False,
            use_psw=True,
            psw_gamma=3.0,
            psw_tau_0=1.0,
            psw_p=2.0,
            psw_use_snr=True,
            use_cpr=True,
            cpr_lambda=0.1,
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/1210",
            action_stride=2,
            pps_weight_path="assets/pace/pps/111_1099_merged/pps_weights_universal.npy",
            assets=AssetsConfig(
                assets_dir="./assets/pi05_tianji_1210_stride2_sft",
            ),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=15_000,
        batch_size=192,
        num_workers=0,
        log_interval=10,
        save_on_steps=(8_000, 15_000),
        keep_period=None,
        fsdp_devices=1,
        data_axis_mode="batch_only",
    ),
    # ---- PACE v5 ablation: PSW without SNR in Psi (flow-time only) -----------
    TrainConfig(
        name="pi05_tianji_pace_v5_no_snr",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            loss_action_dim=16,
            max_token_len=200,
            use_cas=False,
            use_da_fta=False,
            use_pps=False,
            use_psw=True,
            psw_gamma=1.0,
            psw_tau_0=1.0,
            psw_p=2.0,
            psw_use_snr=False,
            use_cpr=True,
            cpr_lambda=0.1,
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/1210",
            action_stride=2,
            pps_weight_path="assets/pace/pps/111_1099_merged/pps_weights_universal.npy",
            assets=AssetsConfig(
                assets_dir="./assets/pi05_tianji_1210_stride2_sft",
            ),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=15_000,
        batch_size=192,
        num_workers=0,
        log_interval=10,
        save_on_steps=(8_000, 15_000),
        keep_period=None,
        fsdp_devices=1,
        data_axis_mode="batch_only",
    ),
    # ---- PACE DA-FTA-Safe + PPS (Phase 3 + 4, legacy) -----------------------
    TrainConfig(
        name="pi05_tianji_da_fta_safe_pps",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            loss_action_dim=16,
            max_token_len=200,
            use_cas=True,
            use_da_fta=True,
            da_fta_sigma_mode="cas",
            use_pps=True,
            cas_init_sigma=(
                0.18539337898682362, 0.21845398819715747, 0.11086866619407716, 0.20364291520733882,
                0.2247156007602922, 0.2431097809096788, 0.48309024536263534, 0.07706123407019662,
                0.15272096818633282, 0.11712216117916195, 0.0963498634092118, 0.07736279768235466,
                0.14435638088184768, 0.26837224316114083, 0.28773935205929574, 0.4504381646454872,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ),
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/0629_438eps_dagge_human_expert",
            action_stride=2,
            pps_weight_path="assets/pace/pps/0629_438eps/pps_weights.npy",
            assets=AssetsConfig(
                assets_dir="./assets/pi05_tianji_joint_0629_438eps_dagge_human_expert_stride2_sft",
            ),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_tianji_joint_0629_438eps_dagge_human_expert_stride2_sft/b192_f1_h15_0629/5000/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=10_000,
        batch_size=192,
        num_workers=0,
        log_interval=10,
        save_interval=1000,
        keep_period=5000,
        fsdp_devices=1,
        data_axis_mode="batch_only",
    ),
    # ---- PACE PPS-only ablation (native MSE + sample weighting) -------------
    TrainConfig(
        name="pi05_tianji_pps_only",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=15,
            loss_action_dim=16,
            max_token_len=200,
            use_cas=False,
            use_pps=True,
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/0629_438eps_dagge_human_expert",
            action_stride=2,
            pps_weight_path="assets/pace/pps/0629_438eps/pps_weights.npy",
            assets=AssetsConfig(
                assets_dir="./assets/pi05_tianji_joint_0629_438eps_dagge_human_expert_stride2_sft",
            ),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "./checkpoints/pi05_tianji_joint_0629_438eps_dagge_human_expert_stride2_sft/b192_f1_h15_0629/5000/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=10_000,
        batch_size=192,
        num_workers=0,
        log_interval=10,
        save_interval=1000,
        keep_period=5000,
        fsdp_devices=1,
        data_axis_mode="batch_only",
    ),
    TrainConfig(
        name="pi05_tianji_joint_20260604_action_expert_sft",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            loss_action_dim=16,
            max_token_len=200,
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/tianji_joint_eef_20260604",
            assets=AssetsConfig(assets_dir="./assets/pi05_tianji_joint_20260604_sft"),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            loss_action_dim=16,
            max_token_len=200,
        ).get_action_expert_freeze_filter(),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=20_000,
        batch_size=32,
        num_workers=4,
        log_interval=50,
        save_interval=1000,
        keep_period=5000,
        fsdp_devices=4,
    ),
    TrainConfig(
        name="pi05_tianji_eef_pose_20260604_sft",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            loss_action_dim=16,
            max_token_len=200,
        ),
        data=LeRobotTianjiEefPoseDataConfig(
            repo_id="local/tianji_R2_eef_pose_20260604",
            assets=AssetsConfig(assets_dir="./assets/pi05_tianji_eef_pose_20260604_sft"),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=20_000,
        batch_size=32,
        num_workers=4,
        log_interval=50,
        save_interval=1000,
        keep_period=5000,
        fsdp_devices=4,
    ),
    TrainConfig(
        name="pi05_tianji_eef_pose_smoke_sft",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            loss_action_dim=16,
            max_token_len=200,
        ),
        data=LeRobotTianjiEefPoseDataConfig(
            repo_id="local/tianji_R2_eef_pose_smoke",
            assets=AssetsConfig(assets_dir="./assets/pi05_tianji_eef_pose_smoke_sft"),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10,
            peak_lr=5e-5,
            decay_steps=100,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=3,
        batch_size=8,
        num_workers=0,
        log_interval=1,
        save_interval=2,
        keep_period=2,
        overwrite=True,
        exp_name="debug",
        wandb_enabled=False,
        fsdp_devices=8,
        data_axis_mode="batch_only",
    ),
    TrainConfig(
        name="pi05_tianji_eef_pose_smoke_lora_sft",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            loss_action_dim=16,
            max_token_len=200,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotTianjiEefPoseDataConfig(
            repo_id="local/tianji_R2_eef_pose_smoke",
            assets=AssetsConfig(assets_dir="./assets/pi05_tianji_eef_pose_smoke_lora_sft"),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            loss_action_dim=16,
            max_token_len=200,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10,
            peak_lr=5e-5,
            decay_steps=100,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=3,
        batch_size=1,
        num_workers=0,
        log_interval=1,
        save_interval=2,
        keep_period=2,
        overwrite=True,
        exp_name="debug",
        wandb_enabled=False,
    ),
    TrainConfig(
        name="pi05_tianji_joint_20260604_sft_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            loss_action_dim=16,
            max_token_len=200,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotTianjiJointDataConfig(
            repo_id="local/tianji_joint_eef_20260604",
            assets=AssetsConfig(assets_dir="./assets/pi05_tianji_joint_20260604_sft"),
            base_config=DataConfig(prompt_from_task=True, video_backend="pyav"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            loss_action_dim=16,
            max_token_len=200,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=20_000,
        batch_size=8,
        num_workers=4,
        log_interval=50,
        save_interval=1000,
        keep_period=5000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

# Candy pick reuses the verified dual-arm PRISM recipe, with independent assets.
_candy_base = next(c for c in _CONFIGS if c.name == "pi05_tianji_pace_v5_462_clean")
_CONFIGS.append(dataclasses.replace(
    _candy_base,
    name="pi05_tianji_candy_581_prism",
    data=dataclasses.replace(
        _candy_base.data,
        repo_id="chenghaoyue/20260914_581eps_candy_pick",
        pps_weight_path="assets/pace/pps/candy_581/pps_weights_universal.npy",
        assets=AssetsConfig(assets_dir="./assets/pi05_tianji_candy_581_stride2_sft"),
    ),
))

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
