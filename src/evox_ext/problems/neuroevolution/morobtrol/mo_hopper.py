import jax.numpy as jnp
from brax.envs.hopper import Hopper


class MoHopper(Hopper):
    """
    The multi-objective version of hopper env (2 objectives).

    :references:
        [1] C. D. Freeman, E. Frey, A. Raichuk, S. Girgin, I. Mordatch, and O. Bachem,
        “Brax - a differentiable physics engine for large scale rigid body simulation,” 2021.
        [Online]. Available: http://github.com/google/brax
    """
    def __init__(self, **kwargs):
        """Initialize the multi-objective hopper env.

        :param num_obj: The number of the objectives. For this env, it is set to 2.
        """
        super().__init__(**kwargs)
        self.num_obj = 2

    def reset(self, rng):
        state = super().reset(rng)
        mo_reward = jnp.zeros((self.num_obj, ))
        return state.replace(reward=mo_reward)

    def step(self, state, action):
        """Run one timestep of the environment's dynamics.

        :param height: The height at which the hopper robot is located.
        For more information, please refer to `hopper <https://github.com/google/brax/tree/main/brax/envs/hopper.py>` env in brax.
        """
        state = super().step(state, action)
        init_z = self.sys.link.transform.pos[0, 2]
        z = state.pipeline_state.x.pos[0, 2]
        height = 10 * (z - init_z)
        mo_reward = jnp.array([state.metrics['reward_forward'], height])
        mo_reward += state.metrics['reward_ctrl']
        mo_reward += state.metrics['reward_healthy']
        return state.replace(reward=mo_reward)
