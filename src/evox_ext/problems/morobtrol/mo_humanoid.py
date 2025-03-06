import jax
import jax.numpy as jnp
from brax.envs.base import State
from brax.envs.humanoid import Humanoid


class MoHumanoid(Humanoid):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_obj = 2

    def reset(self, rng: jax.Array):
        state = super().reset(rng)
        reward = jnp.zeros((self.num_obj,))
        return state.replace(reward=reward)

    def step(self, state: State, action: jax.Array):
        state = super().step(state, action)

        energy_cost = state.metrics["reward_quadctrl"] / self._ctrl_cost_weight
        mo_reward = jnp.array([state.metrics["forward_reward"], energy_cost])
        mo_reward += state.metrics["reward_alive"]
        # state.metrics.update(
        #     cum_reward_alive=state.metrics['cum_reward_alive'] + state.metrics['reward_alive']
        # )
        return state.replace(reward=mo_reward)
