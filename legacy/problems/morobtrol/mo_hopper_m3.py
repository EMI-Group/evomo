from brax.envs.hopper import Hopper
import jax.numpy as jnp


class MoHopper(Hopper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_obj = 3

    def reset(self, rng):
        state = super().reset(rng)
        mo_reward = jnp.zeros((self.num_obj, ))
        return state.replace(reward=mo_reward)

    def step(self, state, action):

        state = super().step(state, action)
        init_z = self.sys.link.transform.pos[0, 2]
        z = state.pipeline_state.x.pos[0, 2]
        height = 10 * (z - init_z)
        energy_cost = state.metrics['reward_ctrl'] / self._ctrl_cost_weight
        mo_reward = jnp.array([state.metrics['reward_forward'], height, energy_cost])
        mo_reward += state.metrics['reward_healthy']
        return state.replace(reward=mo_reward)
