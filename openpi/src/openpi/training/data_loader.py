from collections.abc import Iterator, Sequence
import io
import json
import logging
import multiprocessing
import os
import pathlib
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import datasets
import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
from lerobot.common.datasets.video_utils import decode_video_frames
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch

import openpi.models.model as _model
from openpi.pace.phase_target import PhaseSigmaTargets
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


def _make_action_offsets(action_horizon: int, action_stride: int) -> tuple[int, ...]:
    if action_stride < 1:
        raise ValueError(f"action_stride must be positive, got {action_stride}")
    return tuple(step * action_stride for step in range(action_horizon))


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class _IndexSubsetDataset(Dataset[T_co]):
    """Map a compact training index onto selected indices of the source dataset."""

    def __init__(self, dataset: Dataset[T_co], indices: np.ndarray):
        self._dataset = dataset
        self._indices = np.asarray(indices, dtype=np.int64)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._dataset[int(self._indices[index.__index__()])]

    def __len__(self) -> int:
        return len(self._indices)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


class _LeRobotV3Metadata:
    def __init__(self, root: pathlib.Path):
        self.root = root
        with (root / "meta" / "info.json").open() as f:
            self.info = json.load(f)

        tasks_table = pq.read_table(root / "meta" / "tasks.parquet")
        task_indices = tasks_table["task_index"].to_pylist()
        task_column = next(name for name in tasks_table.column_names if name != "task_index")
        self.tasks = dict(zip(task_indices, tasks_table[task_column].to_pylist(), strict=True))

        episodes_table = pq.read_table(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
        self.episodes = {row["episode_index"]: row for row in episodes_table.to_pylist()}

    @property
    def fps(self) -> int:
        return self.info["fps"]

    @property
    def video_keys(self) -> list[str]:
        return [key for key, feature in self.info["features"].items() if feature["dtype"] == "video"]


class _LeRobotV3Dataset(Dataset):
    """Read local LeRobot v3 merged parquet/video files without rewriting the dataset."""

    def __init__(
        self,
        repo_id: str,
        root: pathlib.Path,
        action_sequence_keys: Sequence[str],
        action_horizon: int,
        action_stride: int,
        video_backend: str | None,
    ):
        self.repo_id = repo_id
        self.root = root
        self.meta = _LeRobotV3Metadata(root)
        self.video_backend = video_backend
        self.action_sequence_keys = tuple(action_sequence_keys)
        self.action_offsets = _make_action_offsets(action_horizon, action_stride)

        self.hf_dataset = datasets.load_dataset(
            "parquet",
            data_files=str(root / self.meta.info["data_path"].format(chunk_index=0, file_index=0)),
            split="train",
        )
        self.hf_dataset.set_transform(lerobot_dataset.hf_transform_to_torch)

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index: SupportsIndex) -> dict:
        idx = index.__index__()
        item = self.hf_dataset[idx]
        episode_index = int(item["episode_index"])
        episode = self.meta.episodes[episode_index]
        episode_end = episode["dataset_to_index"]

        for key in self.action_sequence_keys:
            query_indices = [min(episode_end - 1, idx + offset) for offset in self.action_offsets]
            item[key] = torch.stack(self.hf_dataset.select(query_indices)[key])

        episode_timestamp = float(item["timestamp"])
        for video_key in self.meta.video_keys:
            file_index = episode[f"videos/{video_key}/file_index"]
            chunk_index = episode[f"videos/{video_key}/chunk_index"]
            timestamp_offset = episode[f"videos/{video_key}/from_timestamp"]
            video_path = self.root / self.meta.info["video_path"].format(
                video_key=video_key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            item[video_key] = decode_video_frames(
                video_path,
                [timestamp_offset + episode_timestamp],
                tolerance_s=1e-4,
                backend=self.video_backend,
            ).squeeze(0)

        return item


class _PPSWeightedDataset(Dataset):
    """Attach a precomputed scalar PPS target by the dataset's global frame index."""

    def __init__(self, dataset: Dataset, pps_weight_path: str):
        self.dataset = dataset
        self.pps_weights = np.asarray(np.load(pps_weight_path), dtype=np.float32).reshape(-1)
        if len(self.pps_weights) != len(dataset):
            raise ValueError(
                f"PPS weights length {len(self.pps_weights)} != dataset frames {len(dataset)}"
            )
        if not np.all(np.isfinite(self.pps_weights)):
            raise ValueError(f"PPS weights contain non-finite values: {pps_weight_path}")
        if np.any((self.pps_weights < 0.0) | (self.pps_weights > 1.0)):
            raise ValueError(f"PPS weights must lie in [0, 1]: {pps_weight_path}")
        logging.info(f"PPS weights loaded from {pps_weight_path}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: SupportsIndex) -> dict:
        idx = index.__index__()
        item = dict(self.dataset[idx])
        item["pps_weight"] = torch.tensor(float(self.pps_weights[idx]))
        return item


def _video_cache_dir(repo_id: str) -> pathlib.Path | None:
    """Locate a preprocessed JPEG video cache for `repo_id`, if one exists and looks valid.

    Set OPENPI_VIDEO_CACHE_DIR to override the base directory (default: /root/openpi_video_cache).
    See scripts/preprocess_video_cache.py for how this cache is produced.
    """
    base = pathlib.Path(os.environ.get("OPENPI_VIDEO_CACHE_DIR", "/root/openpi_video_cache"))
    cache_dir = base / repo_id.replace("/", "__")
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        with meta_path.open() as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    for video_key in meta.get("video_keys", []):
        blob = cache_dir / "images" / f"{video_key}.jpegblob"
        offsets = cache_dir / "images" / f"{video_key}.offsets.npy"
        if not (blob.exists() and offsets.exists()):
            return None
    return cache_dir


class _CachedLeRobotV3Dataset(Dataset):
    """Reads a preprocessed JPEG image + numpy cache produced by scripts/preprocess_video_cache.py.

    This avoids on-the-fly video decoding (which was slow and prone to OOM under large
    batch sizes / many dataloader workers) by reading pre-decoded, JPEG-compressed frames
    directly off disk.
    """

    def __init__(
        self,
        repo_id: str,
        cache_dir: pathlib.Path,
        action_sequence_keys: Sequence[str],
        action_horizon: int,
        action_stride: int,
        *,
        skip_images: bool = False,
        cas_asset_dir: str | None = None,
        pps_weight_path: str | None = None,
    ):
        self.repo_id = repo_id
        self.cache_dir = cache_dir
        self._skip_images = skip_images
        with (cache_dir / "meta.json").open() as f:
            self.cache_meta = json.load(f)

        self.total_frames = self.cache_meta["total_frames"]
        self.video_keys = self.cache_meta["video_keys"]
        self.image_size = self.cache_meta["image_size"]
        self.action_sequence_keys = tuple(action_sequence_keys)
        self.action_offsets = _make_action_offsets(action_horizon, action_stride)

        self.state = np.load(cache_dir / "state.npy")
        self.action = np.load(cache_dir / "action.npy")
        self.episode_index = np.load(cache_dir / "episode_index.npy")
        self.timestamp = np.load(cache_dir / "timestamp.npy")
        self.task_index = np.load(cache_dir / "task_index.npy")
        self.episode_to_index = np.load(cache_dir / "episode_to_index.npy")

        # Precompute episode start indices for within-episode frame lookup.
        self.episode_from_index = np.zeros_like(self.episode_to_index)
        if len(self.episode_from_index) > 1:
            self.episode_from_index[1:] = self.episode_to_index[:-1]

        self._phase_targets: PhaseSigmaTargets | None = None
        if cas_asset_dir is not None:
            self._phase_targets = PhaseSigmaTargets(cas_asset_dir)
            logging.info(f"CAS phase targets loaded from {cas_asset_dir}")

        self._pps_weights: np.ndarray | None = None
        if pps_weight_path is not None:
            self._pps_weights = np.asarray(np.load(pps_weight_path), dtype=np.float32).reshape(-1)
            if len(self._pps_weights) != self.total_frames:
                raise ValueError(
                    f"PPS weights length {len(self._pps_weights)} != dataset frames {self.total_frames}"
                )
            if not np.all(np.isfinite(self._pps_weights)):
                raise ValueError(f"PPS weights contain non-finite values: {pps_weight_path}")
            if np.any((self._pps_weights < 0.0) | (self._pps_weights > 1.0)):
                raise ValueError(f"PPS weights must lie in [0, 1]: {pps_weight_path}")
            logging.info(f"PPS weights loaded from {pps_weight_path}")

        self._offsets = {
            video_key: np.load(cache_dir / "images" / f"{video_key}.offsets.npy") for video_key in self.video_keys
        }
        self._blob_paths = {
            video_key: str(cache_dir / "images" / f"{video_key}.jpegblob") for video_key in self.video_keys
        }
        # File descriptors are opened lazily so this dataset is picklable across
        # `multiprocessing.get_context("spawn")` dataloader workers.
        self._blob_fds: dict[str, int] | None = None

    def _fds(self) -> dict[str, int]:
        if self._blob_fds is None:
            self._blob_fds = {video_key: os.open(path, os.O_RDONLY) for video_key, path in self._blob_paths.items()}
        return self._blob_fds

    def __len__(self) -> int:
        return self.total_frames

    def _read_image(self, video_key: str, idx: int) -> torch.Tensor:
        offsets = self._offsets[video_key]
        start, end = int(offsets[idx]), int(offsets[idx + 1])
        data = os.pread(self._fds()[video_key], end - start, start)
        image = Image.open(io.BytesIO(data)).convert("RGB")
        arr = np.asarray(image)  # HWC uint8
        return torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, index: SupportsIndex) -> dict:
        idx = index.__index__()
        episode_idx = int(self.episode_index[idx])
        episode_end = int(self.episode_to_index[episode_idx])
        query_indices = [min(episode_end - 1, idx + offset) for offset in self.action_offsets]

        item: dict = {}
        for key in self.action_sequence_keys:
            item[key] = torch.from_numpy(np.asarray(self.action[query_indices])).float()

        item["observation.state"] = torch.from_numpy(np.asarray(self.state[idx])).float()
        item["episode_index"] = torch.tensor(episode_idx)
        item["timestamp"] = torch.tensor(float(self.timestamp[idx]))
        item["task_index"] = torch.tensor(int(self.task_index[idx]))

        if not self._skip_images:
            for video_key in self.video_keys:
                item[video_key] = self._read_image(video_key, idx)

        if self._phase_targets is not None:
            frame_within_ep = idx - int(self.episode_from_index[episode_idx])
            item["cas_sigma_target"] = torch.from_numpy(
                self._phase_targets.sigma_target(episode_idx, frame_within_ep).copy()
            )

        if self._pps_weights is not None:
            item["pps_weight"] = torch.tensor(float(self._pps_weights[idx]))

        return item


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    skip_images: bool = False,
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    # Match LeRobot's cache-root convention while retaining the historical default.
    # This is useful for mounted datasets that should not be copied into the home cache.
    lerobot_home = pathlib.Path(
        os.environ.get(
            "HF_LEROBOT_HOME",
            pathlib.Path.home() / ".cache" / "huggingface" / "lerobot",
        )
    ).expanduser()
    root = lerobot_home / repo_id
    info_path = root / "meta" / "info.json"
    with info_path.open() as f:
        codebase_version = json.load(f).get("codebase_version")

    pps_attached = False
    episode_indices: np.ndarray | None = None
    if codebase_version == "v3.0":
        dataset_meta = _LeRobotV3Metadata(root)
        cache_dir = _video_cache_dir(repo_id)
        if cache_dir is not None:
            with (cache_dir / "meta.json").open() as f:
                cache_total_frames = json.load(f)["total_frames"]
            expected_total_frames = dataset_meta.info["total_frames"]
            if cache_total_frames != expected_total_frames:
                logging.warning(
                    f"Ignoring stale video cache at {cache_dir} "
                    f"(cache has {cache_total_frames} frames, dataset has {expected_total_frames})"
                )
                cache_dir = None

        if cache_dir is not None:
            logging.info(f"Using preprocessed video cache at {cache_dir}")
            dataset = _CachedLeRobotV3Dataset(
                repo_id,
                cache_dir,
                data_config.action_sequence_keys,
                action_horizon,
                data_config.action_stride,
                skip_images=skip_images,
                cas_asset_dir=data_config.cas_asset_dir,
                pps_weight_path=data_config.pps_weight_path,
            )
            pps_attached = data_config.pps_weight_path is not None
            episode_indices = np.asarray(dataset.episode_index, dtype=np.int64)
        else:
            dataset = _LeRobotV3Dataset(
                repo_id,
                root,
                data_config.action_sequence_keys,
                action_horizon,
                data_config.action_stride,
                data_config.video_backend,
            )
            episode_indices = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
    else:
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
        dataset = lerobot_dataset.LeRobotDataset(
            data_config.repo_id,
            delta_timestamps={
                key: [
                    offset / dataset_meta.fps
                    for offset in _make_action_offsets(action_horizon, data_config.action_stride)
                ]
                for key in data_config.action_sequence_keys
            },
            video_backend=data_config.video_backend,
        )

    if data_config.pps_weight_path is not None and not pps_attached:
        dataset = _PPSWeightedDataset(dataset, data_config.pps_weight_path)

    if data_config.exclude_episode_indices:
        if episode_indices is None:
            raise ValueError("exclude_episode_indices is currently supported only for local LeRobot v3 datasets")
        excluded = np.asarray(data_config.exclude_episode_indices, dtype=np.int64)
        present = set(np.unique(episode_indices).tolist())
        missing = sorted(set(excluded.tolist()) - present)
        if missing:
            raise ValueError(f"Excluded episode indices are absent from {repo_id}: {missing}")
        keep_indices = np.flatnonzero(~np.isin(episode_indices, excluded))
        removed_frames = len(episode_indices) - len(keep_indices)
        logging.info(
            "Episode holdout active for %s: excluded %d episodes and %d frames; %d frames remain",
            repo_id,
            len(excluded),
            removed_frames,
            len(keep_indices),
        )
        dataset = _IndexSubsetDataset(dataset, keep_indices)

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
