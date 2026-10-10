"""JAX Precision-Scaled Weighting kernels for OpenPI/π0.5."""

from __future__ import annotations

import jax.numpy as jnp


def sample_action_scale(actions, *, loss_action_dim: int | None = None):
    """Ground-truth action-chunk RMS, one scalar per sample."""

    active = actions if loss_action_dim is None else actions[..., :loss_action_dim]
    return jnp.sqrt(jnp.mean(jnp.square(active), axis=(-2, -1)))


def sample_action_snr(noise_time, sigma_f, *, eps: float = 1e-6):
    """SNR for OpenPI's convention: 1 is noise and 0 is clean action."""

    return (jnp.square(1.0 - noise_time) * jnp.square(sigma_f)) / (jnp.square(noise_time) + eps)


def psw_weights(
    rho,
    noise_time,
    sigma_f,
    *,
    gamma: float = 2.0,
    tau_0: float = 1.0,
    p: float = 2.0,
    eps: float = 1e-6,
    use_snr: bool = True,
):
    """Return ``1 + gamma * rho * Psi`` for OpenPI flow time."""

    clean_time = 1.0 - noise_time
    if use_snr:
        snr = sample_action_snr(noise_time, sigma_f, eps=eps)
        importance = (1.0 - jnp.exp(-snr / tau_0)) * jnp.power(clean_time, p)
    else:
        importance = jnp.power(clean_time, p)
    return 1.0 + gamma * rho * importance

