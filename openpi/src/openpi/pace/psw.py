"""PACE v5 PSW (Precision-Scaled Weighting) kernels.

See docs/pace_v5_proposal.tex. Training-only; not used at inference.
"""

from __future__ import annotations

import jax.numpy as jnp

from openpi.shared import array_typing as at


@at.typecheck
def sample_action_scale(actions: at.Float[at.Array, "b ah ad"], *, loss_action_dim: int) -> at.Float[at.Array, " b"]:
    """GT action-chunk RMS sigma_f in active model coordinates, one scalar per sample."""
    active = actions[..., :loss_action_dim]
    return jnp.sqrt(jnp.mean(jnp.square(active), axis=(-2, -1)))


@at.typecheck
def sample_action_snr(
    time: at.Float[at.Array, " b"],
    sigma_f: at.Float[at.Array, " b"],
    *,
    eps: float,
) -> at.Float[at.Array, " b"]:
    """Sample action-to-noise ratio with openpi convention: time=1 noise, time=0 clean action."""
    return (jnp.square(1.0 - time) * jnp.square(sigma_f)) / (jnp.square(time) + eps)


@at.typecheck
def precision_importance(
    time: at.Float[at.Array, " b"],
    snr: at.Float[at.Array, " b"],
    *,
    tau_0: float,
    p: float,
) -> at.Float[at.Array, " b"]:
    """Psi(tau, SNR) in [0, 1]: emphasizes low-noise flow times for fine frames."""
    return (1.0 - jnp.exp(-snr / tau_0)) * jnp.power(1.0 - time, p)


@at.typecheck
def psw_weights(
    rho: at.Float[at.Array, " b"],
    time: at.Float[at.Array, " b"],
    sigma_f: at.Float[at.Array, " b"],
    *,
    gamma: float,
    tau_0: float,
    p: float,
    eps: float,
    use_snr: bool = True,
) -> at.Float[at.Array, " b"]:
    """w = 1 + gamma * rho * Psi(tau, SNR), bounded in [1, 1+gamma] for rho in [0, 1]."""
    if use_snr:
        snr = sample_action_snr(time, sigma_f, eps=eps)
        psi = precision_importance(time, snr, tau_0=tau_0, p=p)
    else:
        del sigma_f, tau_0, eps
        psi = jnp.power(1.0 - time, p)
    return 1.0 + gamma * rho * psi
