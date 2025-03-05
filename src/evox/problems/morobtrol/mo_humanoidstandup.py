import jax
import jax.numpy as jnp
from brax.envs.base import State
from brax.envs.humanoidstandup import HumanoidStandup


class MoHumanoidStandup(HumanoidStandup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_obj = 2

    def reset(self, rng: jax.Array):
        state = super().reset(rng)
        reward = jnp.zeros((self.num_obj,))
        return state.replace(reward=reward)

    def step(self, state: State, action: jax.Array):
        state = super().step(state, action)

        quad_energy_cost = state.metrics["reward_quadctrl"]
        mo_reward = jnp.array([state.metrics["reward_linup"], quad_energy_cost])
        # mo_reward += 1
        return state.replace(reward=mo_reward)
