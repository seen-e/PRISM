import contextlib
import logging
from collections.abc import Sequence

import jax
import numpy as np

BATCH_AXIS = "batch"
FSDP_AXIS = "fsdp"
# In FSDP, we shard the data across both the batch and FSDP axes.
DATA_AXIS = (BATCH_AXIS, FSDP_AXIS)
DATA_AXIS_BATCH_ONLY = (BATCH_AXIS,)
DataAxisMode = str


class _MeshState:
    active_mesh: jax.sharding.Mesh | None = None
    data_axis: tuple[str, ...] = DATA_AXIS


def make_mesh(num_fsdp_devices: int) -> jax.sharding.Mesh:
    if jax.device_count() % num_fsdp_devices != 0:
        raise ValueError(
            f"Number of devices {jax.device_count()} must be divisible by the number of FSDP devices {num_fsdp_devices}."
        )
    mesh_shape = (jax.device_count() // num_fsdp_devices, num_fsdp_devices)
    # JAX 0.10 changed make_mesh's default axis type from Auto to Explicit.
    # OpenPI uses with_sharding_constraint for activations, which requires
    # Auto axes and matches the behavior of the older JAX runtime.
    axis_types = (jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto)
    return jax.make_mesh(mesh_shape, (BATCH_AXIS, FSDP_AXIS), axis_types=axis_types)


def data_axis_from_mode(mode: DataAxisMode) -> tuple[str, ...]:
    """Return the mesh axes used for data/activation sharding."""
    match mode:
        case "all":
            return DATA_AXIS
        case "batch_only":
            return DATA_AXIS_BATCH_ONLY
        case _:
            raise ValueError(f"Unknown data axis mode: {mode!r}. Expected 'all' or 'batch_only'.")


def data_partition_spec(mode: DataAxisMode = "all") -> jax.sharding.PartitionSpec:
    return jax.sharding.PartitionSpec(data_axis_from_mode(mode))


@contextlib.contextmanager
def set_mesh(mesh: jax.sharding.Mesh, *, data_axis_mode: DataAxisMode = "all"):
    """Plumbing the mesh deep into the module tree is extremely cumbersome; until the JAX team lands a better API, a
    custom context manager like this one is the recommended way to maintain a reference to a global mesh. This is only used
    in `activation_sharding_constraint` below."""
    if _MeshState.active_mesh is not None:
        raise ValueError("Cannot nest set_mesh context managers.")
    _MeshState.active_mesh = mesh
    _MeshState.data_axis = data_axis_from_mode(data_axis_mode)
    try:
        yield
    finally:
        _MeshState.active_mesh = None
        _MeshState.data_axis = DATA_AXIS


def activation_sharding_constraint(pytree):
    if _MeshState.active_mesh is None:
        return pytree
    return jax.lax.with_sharding_constraint(
        pytree, jax.sharding.NamedSharding(_MeshState.active_mesh, jax.sharding.PartitionSpec(_MeshState.data_axis))
    )


def fsdp_sharding(
    pytree,
    mesh: jax.sharding.Mesh,
    *,
    min_size_mbytes: int = 4,  # 4 MiB
    replicate_embeddings: bool = False,
    log: bool = False,
):
    """Apply FSDP sharding to a pytree of arrays based on the mesh shape.

    Args:
        pytree: A pytree to be apply sharding specified by the mesh, note that only array types (eg. contains .shape attr)
          will be considered for sharding.
        mesh: The mesh being used for applying sharding on to pytree.
        min_size_mbytes: The minimum size of the array in MiB to be considered for sharding, any array smaller than this
          will be replicated.
        log: If true, will log the sharding decisions for arrays that are being considered for sharding.

    Returns:
        The sharded pytree.
    """
    min_size_bytes = min_size_mbytes * 2**20

    def _shard_arr(kp, array: jax.ShapeDtypeStruct):
        # if fsdp is not actually going to be used, replicate everything to avoid extraneous logging
        if mesh.shape[FSDP_AXIS] == 1:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        if replicate_embeddings and _is_embedding_path(kp):
            if log:
                logging.info(f"Replicating embedding-like array {jax.tree_util.keystr(kp)} of shape {array.shape}")
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        # replicate scalar and vector arrays
        if not hasattr(array, "shape"):
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        if len(array.shape) < 2:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        # replicate small arrays
        if (arr_size := np.prod(array.shape) * np.dtype(array.dtype).itemsize) < min_size_bytes:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        # shard matrices and larger tensors along the largest axis that is divisible by the fsdp dimension
        axes = np.argsort(array.shape)[::-1]
        spec = [None] * len(axes)
        for i in axes:
            if array.shape[i] % mesh.shape[FSDP_AXIS] == 0:
                if log:
                    logging.info(
                        f"Sharding {jax.tree_util.keystr(kp)} of shape {array.shape} ({arr_size / 2**20:.2f} MiB) along axis {i}"
                    )
                spec[i] = FSDP_AXIS
                return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))

        # replicate if no valid sharding was found
        if log:
            logging.warning(
                f"Could not find a valid sharding for {jax.tree_util.keystr(kp)} of shape {array.shape} with mesh of shape {mesh.shape}"
            )
        return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    return jax.tree_util.tree_map_with_path(_shard_arr, pytree)


def _is_embedding_path(kp: Sequence[object]) -> bool:
    key = jax.tree_util.keystr(kp)
    return any(
        part in key
        for part in (
            "input_embedding",
            "pos_embedding",
            "embedding",
            "embedder",
        )
    )
