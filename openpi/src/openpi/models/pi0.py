import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
import openpi.pace.psw as _pace_psw
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class ScaleHead(nnx.Module):
    """CAS: Condition-Aware Scale Estimation (legacy PACE v3/v4)."""

    def __init__(
        self,
        in_dim: int,
        action_dim: int,
        hidden_dim: int,
        sigma_min: float,
        sigma_max: float,
        *,
        rngs: nnx.Rngs,
        init_sigma: at.Float[at.Array, " ad"] | None = None,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        init_log_sigma = jnp.log(init_sigma) if init_sigma is not None else jnp.zeros((action_dim,))

        def _bias_init(key, shape, dtype, _v=init_log_sigma):
            del key
            return jnp.broadcast_to(_v, shape).astype(dtype)

        self.fc1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(
            hidden_dim,
            action_dim,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=_bias_init,
            rngs=rngs,
        )

    def __call__(self, h_c: at.Float[at.Array, "b emb"]) -> at.Float[at.Array, "b ad"]:
        x = nnx.swish(self.fc1(h_c))
        log_sigma = self.fc2(x)
        return jnp.clip(jnp.exp(log_sigma), self.sigma_min, self.sigma_max)


class PrecisionHead(nnx.Module):
    """PACE v5 CPR: scalar precision rho_hat(c) from pooled condition embedding."""

    def __init__(self, in_dim: int, hidden_dim: int, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, h_c: at.Float[at.Array, "b emb"]) -> at.Float[at.Array, " b"]:
        x = nnx.swish(self.fc1(h_c))
        return jax.nn.sigmoid(self.fc2(x)[..., 0])


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.loss_action_dim = config.loss_action_dim
        self.image_augmentation = config.image_augmentation
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        self.use_cas = config.use_cas
        if self.use_cas:
            init_sigma = jnp.array(config.cas_init_sigma) if config.cas_init_sigma is not None else None
            self.scale_head = ScaleHead(
                in_dim=paligemma_config.width,
                action_dim=config.action_dim,
                hidden_dim=config.cas_hidden_dim,
                sigma_min=config.cas_sigma_min,
                sigma_max=config.cas_sigma_max,
                init_sigma=init_sigma,
                rngs=rngs,
            )

        self.use_da_fta = config.use_da_fta
        if self.use_da_fta:
            self._da_fta_sigma_mode = config.da_fta_sigma_mode
            self._da_fta_lambda = config.da_fta_lambda
            self._da_fta_kappa = config.da_fta_kappa
            self._da_fta_temperature = config.da_fta_temperature
            self._da_fta_omega_min = config.da_fta_omega_min
            self._da_fta_omega_max = config.da_fta_omega_max
            self._da_fta_eps = config.da_fta_eps
            self._da_fta_global_sigma = (
                jnp.array(config.da_fta_global_sigma) if config.da_fta_global_sigma is not None else None
            )
            self._cas_sigma_reg = config.cas_sigma_reg
        else:
            self._da_fta_sigma_mode = "cas"

        self.use_pps = config.use_pps
        self._pps_alpha = config.pps_alpha

        self.use_psw = config.use_psw
        self._psw_gamma = config.psw_gamma
        self._psw_tau_0 = config.psw_tau_0
        self._psw_p = config.psw_p
        self._psw_eps = config.psw_eps
        self._psw_use_snr = config.psw_use_snr

        self.use_cpr = config.use_cpr
        self._cpr_lambda = config.cpr_lambda
        if self.use_cpr:
            self.precision_head = PrecisionHead(
                in_dim=paligemma_config.width,
                hidden_dim=config.cpr_hidden_dim,
                rngs=rngs,
            )

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @at.typecheck
    def embed_condition(self, obs: _model.Observation) -> at.Float[at.Array, "b emb"]:
        """CAS condition embedding: pooled prefix (vision + language + state)
        representation, independent of the noisy action / flow time tau.

        Reuses the same prefix-only forward pass as the KV-cache warm-up in
        `sample_actions`, so this is cheap relative to a full training step.
        """
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(obs)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), _ = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        mask_f = prefix_mask[..., None].astype(prefix_out.dtype)
        denom = jnp.clip(jnp.sum(mask_f, axis=1), 1e-6, None)
        return jnp.sum(prefix_out * mask_f, axis=1) / denom

    def get_scale(self, obs: _model.Observation) -> at.Float[at.Array, "b ad"]:
        """Returns CAS's predicted per-dimension conditional scale sigma_hat(c)."""
        if not self.use_cas:
            raise ValueError("get_scale() requires the model to be constructed with Pi0Config(use_cas=True).")
        return self.scale_head(self.embed_condition(obs))

    def get_precision(self, obs: _model.Observation) -> at.Float[at.Array, " b"]:
        """Returns CPR's predicted scalar precision demand rho_hat(c)."""
        if not self.use_cpr:
            raise ValueError("get_precision() requires the model to be constructed with Pi0Config(use_cpr=True).")
        return self.precision_head(self.embed_condition(obs))

    def get_precision_and_feature(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, " b"], at.Float[at.Array, "b emb"]]:
        """Returns CPR prediction together with its pooled condition feature."""
        if not self.use_cpr:
            raise ValueError(
                "get_precision_and_feature() requires the model to be constructed with Pi0Config(use_cpr=True)."
            )
        feature = self.embed_condition(obs)
        return self.precision_head(feature), feature

    @at.typecheck
    def compute_cas_warmup_loss(
        self, observation: _model.Observation, sigma_target: at.Float[at.Array, "b ad"]
    ) -> at.Float[at.Array, " b"]:
        """PACE Sec. 19 Phase 2 warm-up loss.

        Regresses log(sigma_hat(c)) towards a self-supervised phase-
        conditional target `sigma_target` (no manual labels; computed
        offline from gripper-closure phase segmentation, see
        `openpi.pace.compute_phase_sigma` / `openpi.pace.phase_target`).

        The backbone should be frozen during this stage via
        `Pi0Config.get_cas_warmup_freeze_filter()` so that only `scale_head`
        receives gradient updates.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        sigma_hat = self.get_scale(observation)
        action_mask = jnp.arange(self.action_dim) < self.loss_action_dim
        log_sq_err = jnp.square(jnp.log(sigma_hat) - jnp.log(jnp.clip(sigma_target, 1e-6, None)))
        return jnp.sum(log_sq_err * action_mask, axis=-1) / self.loss_action_dim

    def _resolve_da_fta_sigma(
        self,
        observation: _model.Observation,
        prefix_out: at.Float[at.Array, "b s emb"],
        prefix_mask: at.Bool[at.Array, "b s"],
    ) -> at.Float[at.Array, "b 1 ad"]:
        """Returns per-dim sigma_hat broadcastable to [B, H, D]."""
        if self._da_fta_sigma_mode == "cas":
            # Detach the context so ScaleHead's regression loss trains only
            # the head itself and never perturbs backbone features.
            prefix_sg = jax.lax.stop_gradient(prefix_out)
            mask_f = prefix_mask[..., None].astype(prefix_sg.dtype)
            denom = jnp.clip(jnp.sum(mask_f, axis=1), 1e-6, None)
            h_c = jnp.sum(prefix_sg * mask_f, axis=1) / denom
            return self.scale_head(h_c)[:, None, :]

        if self._da_fta_sigma_mode == "global":
            return self._da_fta_global_sigma[None, None, :]

        if observation.cas_sigma_target is None:
            raise ValueError(
                'da_fta_sigma_mode="phase_target" requires cas_sigma_target in the batch '
                "(set data.cas_asset_dir)."
            )
        return observation.cas_sigma_target[:, None, :]

    @staticmethod
    def _pool_prefix(
        prefix_out: at.Float[at.Array, "b s emb"], prefix_mask: at.Bool[at.Array, "b s"]
    ) -> at.Float[at.Array, "b emb"]:
        mask_f = prefix_mask[..., None].astype(prefix_out.dtype)
        denom = jnp.clip(jnp.sum(mask_f, axis=1), 1e-6, None)
        return jnp.sum(prefix_out * mask_f, axis=1) / denom

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=train and self.image_augmentation
        )

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        action_mask = jnp.arange(self.action_dim) < self.loss_action_dim

        if self.use_psw:
            squared_error = jnp.square(v_t - u_t)
            per_step_loss = jnp.sum(squared_error * action_mask, axis=-1) / self.loss_action_dim
            if observation.pps_weight is None:
                raise ValueError("use_psw requires pps_weight in the training batch (set pps_weight_path).")
            sigma_f = jax.lax.stop_gradient(
                _pace_psw.sample_action_scale(actions, loss_action_dim=self.loss_action_dim)
            )
            w = _pace_psw.psw_weights(
                observation.pps_weight,
                time,
                sigma_f,
                gamma=self._psw_gamma,
                tau_0=self._psw_tau_0,
                p=self._psw_p,
                eps=self._psw_eps,
                use_snr=self._psw_use_snr,
            )
            per_step_loss = per_step_loss * w[..., None]
        elif self.use_da_fta:
            eps = self._da_fta_eps

            if self._da_fta_sigma_mode == "cas":
                # Conditional scale directly from the GT chunk: per-dim RMS
                # magnitude of the *normalized* actions over the horizon.  In
                # z_t = t*eps + (1-t)*a the per-dim signal power is (1-t)^2 a^2
                # against unit-variance noise, so RMS(a) in normalized units is
                # exactly the sigma the SNR model calls for.  Ground truth is
                # available at training time, so the DA-FTA weight never
                # depends on a learned quantity and cannot degenerate.
                sigma_w = jnp.sqrt(jnp.mean(jnp.square(actions), axis=-2, keepdims=True))  # [B, 1, D]
                sigma_w = jnp.clip(sigma_w, self.scale_head.sigma_min, self.scale_head.sigma_max)
                sigma_w = jax.lax.stop_gradient(sigma_w)
            else:
                sigma_w = jax.lax.stop_gradient(
                    self._resolve_da_fta_sigma(observation, prefix_out, prefix_mask)
                )

            # openpi flow-time convention: time=1 is pure noise, time=0 is the
            # clean action, so the signal coefficient is (1 - time).
            snr_hat = ((1 - time_expanded) ** 2 * sigma_w ** 2) / (time_expanded ** 2 + eps)
            omega = 1.0 + self._da_fta_lambda * jax.nn.sigmoid(
                (jnp.log(snr_hat + eps) - self._da_fta_kappa) / self._da_fta_temperature
            )
            omega = jnp.clip(omega, self._da_fta_omega_min, self._da_fta_omega_max)

            weighted_sq = omega * jnp.square(v_t - u_t)
            per_step_loss = jnp.sum(weighted_sq * action_mask, axis=-1) / self.loss_action_dim

            # CAS ScaleHead: regress log(sigma_hat(c)) onto the same GT chunk
            # RMS.  Training-time DA-FTA weights above never use this head; it
            # exists to predict the conditional scale at inference time (and
            # for analysis).  Context is stop-gradient'ed inside
            # _resolve_da_fta_sigma, so this loss only trains the head.
            if self._da_fta_sigma_mode == "cas" and self._cas_sigma_reg > 0:
                sigma_hat = self._resolve_da_fta_sigma(observation, prefix_out, prefix_mask)
                log_err = jnp.square(jnp.log(sigma_hat[..., 0, :]) - jnp.log(sigma_w[..., 0, :]))
                scale_loss = jnp.sum(log_err * action_mask, axis=-1) / self.loss_action_dim  # [B]
                per_step_loss = per_step_loss + self._cas_sigma_reg * scale_loss[..., None]
        else:
            squared_error = jnp.square(v_t - u_t)
            per_step_loss = jnp.sum(squared_error * action_mask, axis=-1) / self.loss_action_dim

        # Legacy flat PPS (deprecated; use use_psw for PACE v5).
        if self.use_pps and not self.use_psw and observation.pps_weight is not None:
            w_rho = 1.0 + self._pps_alpha * observation.pps_weight[..., None]  # [B, 1]
            per_step_loss = per_step_loss * w_rho

        if self.use_cpr:
            if observation.pps_weight is None:
                raise ValueError("use_cpr requires pps_weight in the training batch (set pps_weight_path).")
            h_c = self._pool_prefix(prefix_out, prefix_mask)
            rho_hat = self.precision_head(h_c)
            cpr_loss = jnp.square(rho_hat - observation.pps_weight)
            per_step_loss = per_step_loss + self._cpr_lambda * cpr_loss[..., None]

        return per_step_loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
