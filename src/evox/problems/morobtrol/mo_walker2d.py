import jax
import jax.numpy as jnp
from brax.envs.base import State
from brax.envs.walker2d import Walker2d


class MoWalker2d(Walker2d):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_obj = 2

    def reset(self, rng):
        state = super().reset(rng)
        mo_reward = jnp.zeros((self.num_obj, ))
        return state.replace(reward=mo_reward)

    def step(self, state: State, action: jax.Array):
        state = super().step(state, action)

        # energy_cost = state.metrics['reward_ctrl']  / self._ctrl_cost_weight
        mo_reward = jnp.array([state.metrics['reward_forward'], state.metrics['reward_ctrl']])
        mo_reward += state.metrics['reward_healthy']

        return state.replace(reward=mo_reward)
