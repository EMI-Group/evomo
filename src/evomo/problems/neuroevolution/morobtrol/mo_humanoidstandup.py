import jax
import jax.numpy as jnp
from brax.envs.base import State
from brax.envs.humanoidstandup import HumanoidStandup


class MoHumanoidStandup(HumanoidStandup):
    """
    The multi-objective version of humanoidstandup.

    :references:
        [1] C. D. Freeman, E. Frey, A. Raichuk, S. Girgin, I. Mordatch, and O. Bachem,
        “Brax - a differentiable physics engine for large scale rigid body simulation,” 2021.
        [Online]. Available: http://github.com/google/brax
    """
    def __init__(self, **kwargs):
        """Initialize the multi-objective humanoidstandup env.

        :param num_obj: The number of the objectives. For this env, it is set to 2.
        """
        super().__init__(**kwargs)
        self.num_obj = 2

    def reset(self, rng: jax.Array):
        state = super().reset(rng)
        reward = jnp.zeros((self.num_obj,))
        return state.replace(reward=reward)

    def step(self, state: State, action: jax.Array):
        """Run one timestep of the environment's dynamics.

        :param quad_energy_cost: The energy consumed by the control force of the humanoidstandup robot.
        For more information, please refer to `humanoidstandup <https://github.com/google/brax/tree/main/brax/envs/humanoidstandup.py>` env in brax.
        """
        state = super().step(state, action)

        quad_energy_cost = state.metrics["reward_quadctrl"]
        mo_reward = jnp.array([state.metrics["reward_linup"], quad_energy_cost])
        return state.replace(reward=mo_reward)
