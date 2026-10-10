from pathlib import Path

import numpy as np
import pytest


def test_pi05_public_config_and_pps_adapter(tmp_path, monkeypatch):
    pytest.importorskip("openpi")
    monkeypatch.setenv("PRISM_REPO_ID", "local/test_dataset")
    monkeypatch.setenv("PRISM_PPS_PATH", str(tmp_path / "pps.npy"))

    from openpi.training import config as openpi_config
    from openpi.training.data_loader import _PPSWeightedDataset

    config = openpi_config.get_config("pi05_prism")
    assert config.model.pi05
    assert config.model.use_psw
    assert config.model.use_cpr

    source = [{"value": 1}, {"value": 2}]
    pps_path = tmp_path / "pps.npy"
    np.save(pps_path, np.array([0.25, 0.75], dtype=np.float32))
    wrapped = _PPSWeightedDataset(source, str(pps_path))
    assert float(wrapped[1]["pps_weight"]) == pytest.approx(0.75)
    assert "pps_weight" not in source[1]


def test_starvla_public_config_and_per_sample_loss(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("starVLA")
    pytest.importorskip("omegaconf")
    monkeypatch.setenv("NO_ALBUMENTATIONS_UPDATE", "1")

    from omegaconf import OmegaConf
    from starVLA.model.framework.share_tools import apply_config_compat
    from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
        LayerwiseFlowmatchingActionHead,
    )

    config_path = Path(__file__).parents[1] / "configs" / "starvla_prism_qwenpi_v3.yaml"
    public = apply_config_compat(OmegaConf.load(config_path))
    assert public.framework.name == "QwenPI_v3"
    assert public.framework.action_model.use_psw
    assert public.framework.action_model.use_cpr

    tiny = OmegaConf.create(
        {
            "framework": {
                "action_model": {
                    "action_dim": 3,
                    "state_dim": 4,
                    "action_horizon": 5,
                    "num_inference_timesteps": 2,
                    "add_pos_embed": True,
                    "max_seq_len": 16,
                    "num_target_vision_tokens": 2,
                    "noise_beta_alpha": 1.5,
                    "noise_beta_beta": 1.0,
                    "noise_s": 0.999,
                    "num_timestep_buckets": 1000,
                    "diffusion_model_cfg": {
                        "num_layers": 2,
                        "input_embedding_dim": 32,
                        "cross_attention_dim": 32,
                        "num_attention_heads": 4,
                        "attention_head_dim": 8,
                        "dropout": 0.0,
                        "final_dropout": False,
                        "interleave_self_attention": True,
                        "norm_type": "ada_norm",
                        "positional_embeddings": None,
                    },
                }
            }
        }
    )
    model = LayerwiseFlowmatchingActionHead(tiny)
    actions = torch.randn(2, 5, 3, requires_grad=True)
    state = torch.randn(2, 1, 4)
    vl_layers = [torch.randn(2, 6, 32) for _ in range(2)]
    attention_mask = torch.ones(2, 6, dtype=torch.bool)
    output = model(
        vl_layers,
        actions,
        state,
        encoder_attention_mask=attention_mask,
        return_details=True,
    )
    assert output["per_sample_loss"].shape == (2,)
    assert output["clean_time"].shape == (2,)
    output["per_sample_loss"].mean().backward()
    assert actions.grad is not None
    assert torch.isfinite(actions.grad).all()
