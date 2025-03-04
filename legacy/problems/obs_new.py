import numpy as np
from functools import partial

import jax
import jax.numpy as jnp


class Obs_Normalizer(object):

    def __init__(
        self,
        observation_shape: int,
        clip_val: float = 5.0,
        std_min: float = 1e-6,
        std_max: float = 1e6,
        useless: bool = False,
    ):
        """Initialization

        Args:
            observation_shape - Shape of the observations.
            std_min - Minimum standard deviation.
            std_max - Maximum standard deviation.
            useless - Judge if this is a useless normalizer, default False.
        """
        self._obs_shape = observation_shape
        self._clip_val = clip_val
        self._std_min = std_min
        self._std_max = std_max
        self._useless = useless

    @partial(jax.jit, static_argnums=(0,))
    def obs_normalize(self, obs: jnp.ndarray, obs_params: jnp.ndarray) -> jnp.ndarray:
        """Normalize the given observation."""

        if self._useless:
            return obs
        else:
            # obss_tp = tuple((self._obs_shape,))
            obs_step = obs_params[0]
            run_var, run_mean = jnp.split(obs_params[1:], 2)
            run_var = run_var.reshape((self._obs_shape,))
            run_mean = run_mean.reshape((self._obs_shape,))
            variance = run_var / (obs_step + 1.0)
            variance = jnp.clip(variance, self._std_min, self._std_max)
            return jnp.clip(
                (obs - run_mean) / jnp.sqrt(variance), -self._clip_val, self._clip_val
            )

    def norm_params_update(
        self, obs_buf: jnp.ndarray, obs_mask: jnp.ndarray, obs_params: jnp.ndarray
    ) -> jnp.ndarray:
        """Update observation normalization parameters."""

        if self._useless:
            return jnp.zeros_like(obs_params)
        else:
            obs_step = obs_params[0]
            run_var, run_mean = jnp.split(obs_params[1:], 2)
            new_mask = obs_mask
            if new_mask.ndim != obs_buf.ndim:
                new_mask = new_mask.reshape(
                    new_mask.shape + (1,) * (obs_buf.ndim - new_mask.ndim)
                )
            new_total_step = obs_step + jnp.sum(new_mask)

            old_mean = (obs_buf - run_mean) * new_mask
            new_mean = run_mean + jnp.sum(old_mean / new_total_step, axis=(0, 1))
            temp_new_mean = (obs_buf - new_mean) * new_mask
            new_var = run_var + jnp.sum(old_mean * temp_new_mean, axis=(0, 1))

            return jnp.concatenate([jnp.ones(1) * new_total_step, new_var, new_mean])

    @partial(jax.jit, static_argnums=(0,))
    def generate_init_params(self) -> jnp.ndarray:
        if self._obs_shape == 0:
            return jnp.zeros(3)
        return jnp.zeros(1 + self._obs_shape * 2)
