"""PyTorch Precision-Scaled Weighting kernels for StarVLA."""

from __future__ import annotations

import torch


def sample_action_scale(actions: torch.Tensor, *, loss_action_dim: int | None = None) -> torch.Tensor:
    """Ground-truth action-chunk RMS, one scalar per sample."""

    active = actions if loss_action_dim is None else actions[..., :loss_action_dim]
    return active.float().square().mean(dim=(-2, -1)).sqrt().to(actions.dtype)


def sample_action_snr(clean_time: torch.Tensor, sigma_f: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """SNR for StarVLA's convention: 0 is noise and 1 is clean action."""

    return clean_time.square() * sigma_f.square() / ((1.0 - clean_time).square() + eps)


def psw_weights(
    rho: torch.Tensor,
    clean_time: torch.Tensor,
    sigma_f: torch.Tensor,
    *,
    gamma: float = 2.0,
    tau_0: float = 1.0,
    p: float = 2.0,
    eps: float = 1e-6,
    use_snr: bool = True,
) -> torch.Tensor:
    """Return ``1 + gamma * rho * Psi`` for StarVLA flow time."""

    if use_snr:
        snr = sample_action_snr(clean_time, sigma_f, eps=eps)
        importance = (1.0 - torch.exp(-snr / tau_0)) * clean_time.pow(p)
    else:
        importance = clean_time.pow(p)
    return 1.0 + gamma * rho * importance

