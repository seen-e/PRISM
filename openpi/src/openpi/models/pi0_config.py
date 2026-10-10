import dataclasses
from typing import TYPE_CHECKING, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    # Number of leading action dimensions that contribute to the flow-matching loss.
    # This allows datasets with smaller action spaces to use the pretrained model's
    # fixed action_dim without supervising zero-padded dimensions.
    loss_action_dim: int | None = None
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    # Can be disabled for runtimes where augmax does not support the active
    # explicit mesh/sharding implementation.
    image_augmentation: bool = True

    pytorch_compile_mode: str | None = "max-autotune"

    # --- PACE / CAS (Condition-Aware Scale Estimation) ---------------------
    # See PACE/proposal.md Sec. 19, Phase 2. Additive and off by default:
    # setting use_cas=False reproduces stock pi0.5 exactly (no extra params,
    # no extra forward pass).
    use_cas: bool = False
    cas_hidden_dim: int = 256
    cas_sigma_min: float = 1e-3
    cas_sigma_max: float = 2.0
    # Optional per-dimension init for log(sigma), e.g. global_sigma_init from
    # openpi.pace.compute_phase_sigma output. If None, the ScaleHead's last
    # layer is bias-initialized to log(1.0) for every dimension.
    cas_init_sigma: tuple[float, ...] | None = None

    # --- PACE / DA-FTA-Safe (Phase 3) loss hyperparameters ---
    # See proposal.md Sec. 19, Phase 3.  Independent of CAS when sigma comes
    # from a fixed global or offline phase target instead of ScaleHead.
    use_da_fta: bool = False
    # Where per-dimension sigma for DA-FTA comes from:
    #   "cas"           -> ScaleHead(h_c), requires use_cas=True
    #   "global"        -> fixed da_fta_global_sigma (dataset stats)
    #   "phase_target"  -> observation.cas_sigma_target from offline precompute
    da_fta_sigma_mode: Literal["cas", "global", "phase_target"] = "cas"
    # Per-dimension global sigma for da_fta_sigma_mode="global" (e.g. from
    # phase_sigma.json global_sigma_init or norm stats).
    da_fta_global_sigma: tuple[float, ...] | None = None
    da_fta_lambda: float = 1.0
    da_fta_kappa: float = 0.0
    da_fta_temperature: float = 1.0
    da_fta_omega_min: float = 0.1
    da_fta_omega_max: float = 10.0
    da_fta_eps: float = 1e-6
    # Weight of the CAS ScaleHead regression loss: log(sigma_hat(c)) is
    # regressed onto the per-sample, per-dim RMS of the normalized GT action
    # chunk.  The head is inference-only; training-time DA-FTA weights are
    # computed from the GT chunk directly and never depend on the head.
    cas_sigma_reg: float = 0.1

    # --- PACE / PPS (Phase 4, legacy flat weighting) -----------------------
    # Deprecated for new runs: use use_psw (PACE v5) instead.
    use_pps: bool = False
    pps_alpha: float = 1.0

    # --- PACE v5: PSW + CPR (training only; inference unchanged) -----------
    use_psw: bool = False
    psw_gamma: float = 1.0
    psw_tau_0: float = 1.0
    psw_p: float = 2.0
    psw_eps: float = 1e-6
    # If False, Psi uses only flow time (1-tau)^p (SNR ablation).
    psw_use_snr: bool = True
    use_cpr: bool = False
    cpr_lambda: float = 0.1
    cpr_hidden_dim: int = 256

    def __post_init__(self):
        if self.loss_action_dim is None:
            object.__setattr__(self, "loss_action_dim", self.action_dim)
        if not 1 <= self.loss_action_dim <= self.action_dim:
            raise ValueError(
                f"loss_action_dim must be in [1, action_dim], got {self.loss_action_dim} for action_dim={self.action_dim}"
            )
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]
        if self.cas_init_sigma is not None and len(self.cas_init_sigma) != self.action_dim:
            raise ValueError(
                f"cas_init_sigma must have length action_dim={self.action_dim}, got {len(self.cas_init_sigma)}"
            )
        if self.use_psw or self.use_cpr:
            if self.use_cas or self.use_da_fta:
                raise ValueError("PACE v5 (use_psw/use_cpr) cannot be combined with legacy use_cas/use_da_fta.")
            if self.use_pps:
                raise ValueError("PACE v5 uses use_psw for rho weighting; disable legacy use_pps.")
        if not self.use_da_fta:
            return
        if self.da_fta_sigma_mode == "cas" and not self.use_cas:
            raise ValueError('da_fta_sigma_mode="cas" requires use_cas=True.')
        if self.da_fta_sigma_mode == "global":
            if self.da_fta_global_sigma is None:
                raise ValueError('da_fta_sigma_mode="global" requires da_fta_global_sigma.')
            if len(self.da_fta_global_sigma) != self.action_dim:
                raise ValueError(
                    f"da_fta_global_sigma must have length action_dim={self.action_dim}, "
                    f"got {len(self.da_fta_global_sigma)}"
                )

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)

    def get_action_expert_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze the VLM and vision tower while fully training the action expert."""
        action_expert_filter = nnx.Any(
            nnx_utils.PathRegex(".*llm.*_1.*"),
            nnx_utils.PathRegex(
                ".*(action_in_proj|action_out_proj|state_proj|"
                "action_time_mlp_in|action_time_mlp_out|time_mlp_in|time_mlp_out).*"
            ),
        )
        return nnx.Not(action_expert_filter)

    def get_cas_warmup_freeze_filter(self) -> nnx.filterlib.Filter:
        """PACE Sec. 19 Phase 2: freeze everything except `scale_head`.

        Used during CAS warm-up so the not-yet-trained ScaleHead does not
        inject noisy gradients into the already-deployed backbone. Only
        valid when `use_cas=True`.
        """
        if not self.use_cas:
            raise ValueError("get_cas_warmup_freeze_filter() requires Pi0Config(use_cas=True).")
        return nnx.Not(nnx_utils.PathRegex(".*scale_head.*"))
