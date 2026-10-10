import numpy as np
import pytest

from prism.data import EpisodePPSLookup, PPSWeightedDataset
from prism.lerobot import compute_pps_records
from prism.pps import PPSConfig, compute_episode_pps


def test_pps_is_finite_and_bounded():
    state = np.zeros((32, 22), dtype=np.float32)
    state[:, :14] = np.linspace(0.0, 0.2, 32)[:, None]
    state[:, 14:20] = np.linspace(0.0, 0.1, 32)[:, None]
    state[:, 20:22] = np.linspace(0.0, 1.0, 32)[:, None]
    rho = compute_episode_pps(state, PPSConfig())
    assert rho.shape == (32,)
    assert np.all(np.isfinite(rho))
    assert np.all((0.0 <= rho) & (rho <= 1.0))


def test_stationary_episode_is_neutral():
    rho = compute_episode_pps(np.zeros((8, 22), dtype=np.float32))
    np.testing.assert_allclose(rho, 0.5)


def test_dataset_adapter_does_not_mutate_source():
    source = [{"x": 1}, {"x": 2}]
    wrapped = PPSWeightedDataset(source, [0.25, 0.75])
    assert wrapped[1]["pps_weight"] == np.float32(0.75)
    assert "pps_weight" not in source[1]


def test_episode_lookup_uses_frame_identity(tmp_path):
    np.savez_compressed(
        tmp_path / "demo.npz",
        trajectory_id=np.array([3, 3, 7]),
        frame_index=np.array([0, 1, 0]),
        pps_weight=np.array([0.2, 0.8, 0.4]),
    )
    lookup = EpisodePPSLookup(tmp_path)
    assert lookup("demo", 3, 1) == np.float32(0.8)


def test_record_generation_is_episode_local_and_globally_sorted():
    state = np.zeros((6, 22), dtype=np.float32)
    state[:, :14] = np.arange(6)[:, None] * 0.01
    episode = np.array([1, 0, 1, 0, 1, 0])
    frame = np.array([1, 0, 0, 2, 2, 1])
    global_index = np.array([4, 0, 3, 2, 5, 1])
    rho, sorted_episode, sorted_frame = compute_pps_records(
        state,
        episode,
        frame,
        global_index=global_index,
    )
    assert rho.shape == (6,)
    np.testing.assert_array_equal(sorted_episode, [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(sorted_frame, [0, 1, 2, 0, 1, 2])


def test_torch_and_jax_psw_agree():
    jnp = pytest.importorskip("jax.numpy")
    torch = pytest.importorskip("torch")

    from prism.psw_jax import psw_weights as jax_weights
    from prism.psw_torch import psw_weights as torch_weights

    rho = np.array([0.0, 0.4, 1.0], dtype=np.float32)
    noise_time = np.array([0.8, 0.5, 0.2], dtype=np.float32)
    sigma = np.array([0.6, 1.0, 1.4], dtype=np.float32)
    actual_jax = np.asarray(jax_weights(jnp.asarray(rho), jnp.asarray(noise_time), jnp.asarray(sigma)))
    actual_torch = torch_weights(
        torch.from_numpy(rho),
        torch.from_numpy(1.0 - noise_time),
        torch.from_numpy(sigma),
    ).numpy()
    np.testing.assert_allclose(actual_jax, actual_torch, rtol=1e-5, atol=1e-6)


def test_torch_psw_is_bounded_and_differentiable():
    torch = pytest.importorskip("torch")
    from prism.psw_torch import psw_weights, sample_action_scale

    actions = torch.randn(3, 5, 4, requires_grad=True)
    rho = torch.tensor([0.0, 0.5, 1.0])
    clean_time = torch.tensor([0.2, 0.5, 0.8])
    sigma = sample_action_scale(actions)
    weights = psw_weights(rho, clean_time, sigma, gamma=2.0)
    assert torch.all(weights >= 1.0)
    assert torch.all(weights <= 3.0)
    weights.sum().backward()
    assert actions.grad is not None
    assert torch.isfinite(actions.grad).all()
