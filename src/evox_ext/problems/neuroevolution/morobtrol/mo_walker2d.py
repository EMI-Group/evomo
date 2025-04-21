import jax
import jax.numpy as jnp
from brax.envs.base import State
from brax.envs.walker2d import Walker2d


class MoWalker2d(Walker2d):
    """
    The multi-objective version of walker2d env.

    :references:
        [1] C. D. Freeman, E. Frey, A. Raichuk, S. Girgin, I. Mordatch, and O. Bachem,
        “Brax - a differentiable physics engine for large scale rigid body simulation,” 2021.
        [Online]. Available: http://github.com/google/brax
    """
    def __init__(self, **kwargs):
        """Initialize the multi-objective walker2d env.

        :param num_obj: The number of the objectives. For this env, it is set to 2.
        """
        super().__init__(**kwargs)
        self.num_obj = 2

    def reset(self, rng):
        state = super().reset(rng)
        mo_reward = jnp.zeros((self.num_obj, ))
        return state.replace(reward=mo_reward)

    def step(self, state: State, action: jax.Array):
        """Run one timestep of the environment's dynamics.

        For more information, please refer to `walker2d <https://github.com/google/brax/tree/main/brax/envs/walker2d.py>` env in brax.
        """
        state = super().step(state, action)
        mo_reward = jnp.array([state.metrics['reward_forward'], state.metrics['reward_ctrl']])
        mo_reward += state.metrics['reward_healthy']

        return state.replace(reward=mo_reward)
