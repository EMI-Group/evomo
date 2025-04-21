import jax.numpy as jnp
from brax.envs.reacher import Reacher


class MoReacher(Reacher):
    """
    The multi-objective version of reacher env.

    :references:
        [1] C. D. Freeman, E. Frey, A. Raichuk, S. Girgin, I. Mordatch, and O. Bachem,
        “Brax - a differentiable physics engine for large scale rigid body simulation,” 2021.
        [Online]. Available: http://github.com/google/brax
    """
    def __init__(self, **kwargs):
        """Initialize the multi-objective reacher env.

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

        For more information, please refer to `reacher <https://github.com/google/brax/tree/main/brax/envs/reacher.py>` env in brax.
        """
        state = super().step(state, action)
        mo_reward = jnp.array(
            [
                state.metrics["reward_dist"],
                state.metrics["reward_ctrl"],
            ]
        )
        return state.replace(reward=mo_reward)
