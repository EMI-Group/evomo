import jax.numpy as jnp
from brax.envs.pusher import Pusher


class MoPusher(Pusher):
    def __init__(self, cost_obj=None, **kwargs):
        super().__init__(**kwargs)
        self.num_obj = 3

    def reset(self, rng):
        state = super().reset(rng)
        mo_reward = jnp.zeros((self.num_obj,))
        return state.replace(reward=mo_reward)

    def step(self, state, action):
        state = super().step(state, action)
        mo_reward = jnp.array(
            [
                state.metrics["reward_near"],
                state.metrics["reward_dist"],
                state.metrics["reward_ctrl"],
            ]
        )
        return state.replace(reward=mo_reward)
