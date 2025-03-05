import jax.numpy as jnp
from brax import base
from brax.envs.inverted_double_pendulum import InvertedDoublePendulum


class MoInvertedDoublePendulum(InvertedDoublePendulum):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_obj = 2

    def reset(self, rng):
        state = super().reset(rng)
        mo_reward = jnp.zeros((self.num_obj,))
        return state.replace(reward=mo_reward)

    # def step(self, state, action):
    #     state = super().step(state, action)
    #     mo_reward = jnp.array([state.metrics['reward_run'], state.metrics['reward_ctrl']])
    #     return state.replace(reward=mo_reward)

    def step(self, state, action):
        """Run one timestep of the environment's dynamics."""
        pipeline_state = self.pipeline_step(state.pipeline_state, action)

        tip = base.Transform.create(pos=jnp.array([0.0, 0.0, 0.6])).do(pipeline_state.x.take(2))
        x, _, y = tip.pos
        dist_penalty = 0.01 * x**2 + (y - 2) ** 2
        v1, v2 = pipeline_state.qd[1:]
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        alive_bonus = 10

        obs = self._get_obs(pipeline_state)
        # reward = alive_bonus - dist_penalty - vel_penalty
        done = jnp.where(y <= 1, jnp.float32(1), jnp.float32(0))

        mo_reward = jnp.array([alive_bonus - dist_penalty, alive_bonus - vel_penalty])
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=mo_reward, done=done)
