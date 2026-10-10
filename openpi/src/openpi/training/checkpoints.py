from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
from typing import Protocol

from etils import epath
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str,
    *,
    keep_period: int | None,
    max_to_keep: int | None = 1,
    overwrite: bool,
    resume: bool,
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    # Only persist params + assets (not full train_state with optimizer state).
    # This matches SFT checkpoints and keeps host-RAM spikes during save manageable.
    # Inference/policy loading only needs params/ and assets/.
    with at.disable_typechecking():
        _, params = _split_params(state)
    items = {
        "assets": save_assets,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)
    checkpoint_manager.wait_until_finished()


def _checkpoint_has_train_state(checkpoint_manager: ocp.CheckpointManager, step: int) -> bool:
    return (epath.Path(checkpoint_manager.directory) / str(step) / "train_state").exists()


def has_train_state(checkpoint_manager: ocp.CheckpointManager, step: int) -> bool:
    """Returns whether a checkpoint contains optimizer state."""
    return _checkpoint_has_train_state(checkpoint_manager, step)


def advance_lightweight_state(state: training_utils.TrainState, step: int) -> training_utils.TrainState:
    """Advance counters after params were loaded directly during initialization."""
    def advance_opt_count(path, value):
        if path and getattr(path[-1], "name", None) == "count":
            return value + jnp.asarray(step, dtype=value.dtype)
        return value

    return dataclasses.replace(
        state,
        step=jnp.asarray(step, dtype=state.step.dtype),
        opt_state=jax.tree_util.tree_map_with_path(advance_opt_count, state.opt_state),
    )


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restore_step = step if step is not None else checkpoint_manager.latest_step()
        items: dict[str, object] = {"params": {"params": params}}
        if _checkpoint_has_train_state(checkpoint_manager, restore_step):
            items["train_state"] = train_state
        restored = checkpoint_manager.restore(step, items=items)
    if "train_state" in restored:
        return _merge_params(restored["train_state"], restored["params"])
    # Lightweight checkpoints intentionally omit optimizer moments.  Restore
    # the saved EMA/inference weights as both live and EMA parameters, advance
    # every Optax counter to the checkpoint step, and start with zero moments.
    # This is the closest valid continuation possible from a params-only save.
    restored_params = restored["params"]["params"]
    # Explicit copy is required because train_step donates TrainState.  Algebraic
    # copies such as x + 0 are folded away by XLA and make params/EMA alias.
    live_params = jax.tree.map(jnp.copy, restored_params)
    # Adam and the LR schedule each own a scalar field named ``count``.  Do
    # not tree_set both to the same Array because donated TrainState then has
    # an aliased scalar buffer.  Advance each existing counter independently.
    def advance_opt_count(path, value):
        if path and getattr(path[-1], "name", None) == "count":
            return value + jnp.asarray(restore_step, dtype=value.dtype)
        return value

    opt_state = jax.tree_util.tree_map_with_path(advance_opt_count, train_state.opt_state)
    train_state = dataclasses.replace(
        train_state,
        step=jnp.asarray(restore_step, dtype=train_state.step.dtype),
        params=live_params,
        ema_params=restored_params if train_state.ema_decay is not None else None,
        opt_state=opt_state,
    )
    logging.warning(
        "Resumed lightweight checkpoint at step %s: restored EMA params as live params; "
        "optimizer moments were unavailable and were reinitialized with counters advanced.",
        restore_step,
    )
    return train_state


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
