import jax.numpy as jnp
from brax import base
from brax.envs.inverted_double_pendulum import InvertedDoublePendulum


class MoInvertedDoublePendulum(InvertedDoublePendulum):
    """
    The multi-objective version of inverted_double_pendulum env.

    :references:
        [1] C. D. Freeman, E. Frey, A. Raichuk, S. Girgin, I. Mordatch, and O. Bachem,
        “Brax - a differentiable physics engine for large scale rigid body simulation,” 2021.
        [Online]. Available: http://github.com/google/brax
    """
    def __init__(self, **kwargs):
        """Initialize the multi-objective inverted_double_pendulum env.

        :param num_obj: The number of the objectives. For this env, it is set to 2.
        """
        super().__init__(**kwargs)
        self.num_obj = 2

    def reset(self, rng):
        state = super().reset(rng)
        mo_reward = jnp.zeros((self.num_obj,))
        return state.replace(reward=mo_reward)

    def step(self, state, action):
        """Run one timestep of the environment's dynamics.

        For more information, please refer to `inverted_double_pendulum <https://github.com/google/brax/tree/main/brax/envs/inverted_double_pendulum.py>` env in brax.
        """
        pipeline_state = self.pipeline_step(state.pipeline_state, action)

        tip = base.Transform.create(pos=jnp.array([0.0, 0.0, 0.6])).do(pipeline_state.x.take(2))
        x, _, y = tip.pos
        dist_penalty = 0.01 * x**2 + (y - 2) ** 2
        v1, v2 = pipeline_state.qd[1:]
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        alive_bonus = 10

        obs = self._get_obs(pipeline_state)
        done = jnp.where(y <= 1, jnp.float32(1), jnp.float32(0))

        mo_reward = jnp.array([alive_bonus - dist_penalty, alive_bonus - vel_penalty])
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=mo_reward, done=done)
