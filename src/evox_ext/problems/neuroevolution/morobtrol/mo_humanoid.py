import jax
import jax.numpy as jnp
from brax.envs.base import State
from brax.envs.humanoid import Humanoid


class MoHumanoid(Humanoid):
    """
    The multi-objective version of humanoid env.

    :references:
        [1] C. D. Freeman, E. Frey, A. Raichuk, S. Girgin, I. Mordatch, and O. Bachem,
        “Brax - a differentiable physics engine for large scale rigid body simulation,” 2021.
        [Online]. Available: http://github.com/google/brax
    """
    def __init__(self, **kwargs):
        """Initialize the multi-objective humanoid env.

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

        :param energy_cost: The energy consumed by the humanoid robot.
        For more information, please refer to `humanoid <https://github.com/google/brax/tree/main/brax/envs/humanoid.py>` env in brax.
        """
        state = super().step(state, action)

        energy_cost = state.metrics["reward_quadctrl"] / self._ctrl_cost_weight
        mo_reward = jnp.array([state.metrics["forward_reward"], energy_cost])
        mo_reward += state.metrics["reward_alive"]
        return state.replace(reward=mo_reward)
