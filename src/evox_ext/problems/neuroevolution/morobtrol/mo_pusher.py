import jax.numpy as jnp
from brax.envs.pusher import Pusher


class MoPusher(Pusher):
    """
    The multi-objective version of pusher env.

    :references:
        [1] C. D. Freeman, E. Frey, A. Raichuk, S. Girgin, I. Mordatch, and O. Bachem,
        “Brax - a differentiable physics engine for large scale rigid body simulation,” 2021.
        [Online]. Available: http://github.com/google/brax
    """
    def __init__(self, **kwargs):
        """Initialize the multi-objective pusher env.

        :param num_obj: The number of the objectives. For this env, it is set to 3.
        """
        super().__init__(**kwargs)
        self.num_obj = 3

    def reset(self, rng):
        state = super().reset(rng)
        mo_reward = jnp.zeros((self.num_obj,))
        return state.replace(reward=mo_reward)

    def step(self, state, action):
        """Run one timestep of the environment's dynamics.

        For more information, please refer to `pusher <https://github.com/google/brax/tree/main/brax/envs/pusher.py>` env in brax.
        """
        state = super().step(state, action)
        mo_reward = jnp.array(
            [
                state.metrics["reward_near"],
                state.metrics["reward_dist"],
                state.metrics["reward_ctrl"],
            ]
        )
        return state.replace(reward=mo_reward)
