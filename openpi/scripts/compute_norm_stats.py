"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def _is_tianji_joint_repo(repo_id: str | None) -> bool:
    if repo_id is not None and repo_id.startswith(("local/robotwin_", "local/robotwin50_")):
        return True
    return repo_id in {
        "local/tianji_joint_eef_20260523",
        "local/tianji_joint_eef_20260604",
        "local/tianji_joint_537_20260610",
        "local/tianji_joint_504_newop_20260612",
        "local/tianji_R2_eef_pose_20260604",
        "local/tianji_R2_eef_pose_smoke",
        "local/0629_438eps_dagge_human_expert",
        "local/1210",
        "local/rlbench_pace_precision",
        "local/rlbench10_pace",
        "local/robotwin_pace_precision",
    }


def _disable_video_decoding_for_stats(dataset: _data_loader.Dataset, data_config: _config.DataConfig) -> None:
    if not _is_tianji_joint_repo(data_config.repo_id):
        return
    inner = getattr(dataset, "_dataset", None)
    if inner is not None:
        _disable_video_decoding_for_stats(inner, data_config)
    meta = getattr(dataset, "meta", None)
    if meta is None:
        return
    for key in list(meta.video_keys):
        meta.info["features"].pop(key, None)


def _repack_inputs_for_stats(data_config: _config.DataConfig) -> list[transforms.DataTransformFn]:
    if _is_tianji_joint_repo(data_config.repo_id):
        return [
            transforms.RepackTransform(
                {
                    "state": "observation.state",
                    "actions": "action",
                }
            )
        ]
    return list(data_config.repack_transforms.inputs)


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(
        data_config, action_horizon, model_config, skip_images=True
    )
    _disable_video_decoding_for_stats(dataset, data_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *_repack_inputs_for_stats(data_config),
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
