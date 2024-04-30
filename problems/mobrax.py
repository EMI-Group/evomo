from typing import Callable, Any

# from brax import envs
# from mo_brax import envs
import brax.envs as brax_envs

# import mo_brax.envs as mo_brax_envs
from brax.io import html, image
import jax
from jax import jit, vmap
import jax.numpy as jnp
from jax.tree_util import tree_leaves
from evox import Problem, State, jit_method
from jax.experimental.host_callback import id_print
from .obs_new import Obs_Normalizer


class MoBrax(Problem):
    def __init__(
        self,
        policy: Callable,
        env_name: str,
        cap_episode: int,
        obs_norm: Obs_Normalizer = None,
        backend: str = "generalized",
        num_obj: int = 1,
    ):
        """Contruct a brax-based problem

        Parameters
        ----------
        policy
            A function that accept two arguments
            the first one is the parameter and the second is the input.
        env_name
            The environment name.
        batch_size
            The number of brax environments to run in parallel.
            Usually this should match the population size at the algorithm side.
        cap_episode
            The maximum number episodes to run.
        backend
            Brax's backend, one of "generalized", "positional", "spring".
            Default to "generalized".
        """
        self.batched_policy = jit(vmap(policy))
        self.policy = policy
        self.env_name = env_name
        self.backend = backend
        self.env = brax_envs.wrappers.training.VmapWrapper(
            brax_envs.get_environment(env_name=env_name, backend=backend)
        )
        self.num_obj = num_obj
        self.cap_episode = cap_episode
        if obs_norm is None:
            self.obs_norm = Obs_Normalizer(observation_shape=0, useless=True)
        else:
            self.obs_norm = obs_norm
        self.jit_reset = jit(self.env.reset)
        self.jit_env_step = jit(self.env.step)

    def setup(self, key):
        return State(key=key, obs_param=self.obs_norm.generate_init_params())

    # @jit_method
    # def evaluate(self, state, weights):
    #     batch_size = tree_leaves(weights)[0].shape[0]
    #     brax_state = self.jit_reset(jnp.tile(state.key, (batch_size, 1)))
    #
    #     def cond_func(val):
    #         counter, state, _total_reward = val
    #         return (counter < self.cap_episode) & (~state.done.all())
    #
    #     def body_func(val):
    #         counter, brax_state, total_reward = val
    #         action = self.batched_policy(weights, brax_state.obs)
    #         brax_state = self.jit_env_step(brax_state, action)
    #         id_print(brax_state.done)
    #         total_reward += (1 - brax_state.done) * brax_state.reward
    #         return counter + 1, brax_state, total_reward
    #
    #     init_val = (0, brax_state, jnp.zeros((batch_size, 2)))  #　self.num_obj
    #
    #     _counter, _brax_state, total_reward = jax.lax.while_loop(
    #         cond_func, body_func, init_val
    #     )
    #     return total_reward, state
    @jit_method
    def evaluate(self, state, weights):
        batch_size = tree_leaves(weights)[0].shape[0]
        # print(obs_sha)
        brax_state = self.jit_reset(jnp.tile(state.key, (batch_size, 1)))

        def cond_func(val):
            counter, state, _total_reward, _, _, _, _ = val
            return (counter < self.cap_episode) & (~state.done.all())

        def body_func(val):
            (
                counter,
                brax_state,
                total_reward,
                batch_step,
                valid_mask,
                obs_buf,
                origin_obs,
            ) = val

            origin_obs = brax_state.obs
            norm_obs = self.obs_norm.obs_normalize(origin_obs, state.obs_param)
            brax_state = brax_state.replace(obs=norm_obs)

            action = self.batched_policy(weights, brax_state.obs)
            brax_state = self.jit_env_step(brax_state, action)
            # obs_buf.append(brax_state.obs)
            obs_buf = obs_buf.at[counter].set(brax_state.obs)
            batch_step += 1 - brax_state.done
            done = jnp.tile(brax_state.done[:, jnp.newaxis], (1, self.num_obj))
            reward = jnp.nan_to_num(brax_state.reward)
            total_reward += (1 - done) * reward
            # valid_mask = (1 - brax_state.done.ravel()) * valid_mask
            valid_mask = valid_mask.at[counter].set(
                (1 - brax_state.done.ravel()) * valid_mask[counter]
            )
            return (
                counter + 1,
                brax_state,
                total_reward,
                batch_step,
                valid_mask,
                obs_buf,
                origin_obs,
            )

        valid_mask = jnp.ones((self.cap_episode, batch_size))
        obs_shape = self.obs_norm._obs_shape
        obs_buf = jnp.zeros((self.cap_episode, batch_size, obs_shape))
        origin_obs = brax_state.obs
        init_val = (
            0,
            brax_state,
            jnp.zeros((batch_size, self.num_obj)),
            jnp.zeros((batch_size,)),
            valid_mask,
            obs_buf,
            origin_obs,
        )

        (
            _counter,
            _brax_state,
            total_reward,
            batch_step,
            valid_mask,
            obs_buf,
            origin_obs,
        ) = jax.lax.while_loop(cond_func, body_func, init_val)

        if not self.obs_norm._useless:
            state.update(
                obs_param=self.obs_norm.norm_params_update(
                    obs_buf=obs_buf, obs_mask=valid_mask, obs_params=state.obs_param
                )
            )

        return total_reward, state

    # @jit_method
    # def evaluate_single(self, state, weights):
    #     batch_size = tree_leaves(weights)[0].shape[0]
    #     brax_state = self.jit_reset(jnp.tile(state.key, (batch_size, 1)))
    #
    #     def cond_func(val):
    #         counter, state, _total_reward, _ = val
    #         return (counter < self.cap_episode) & (~state.done.all())
    #
    #     def body_func(val):
    #         counter, brax_state, total_reward, batch_step = val
    #         action = self.batched_policy(weights, brax_state.obs)
    #         brax_state = self.jit_env_step(brax_state, action)
    #         batch_step += (1 - brax_state.done)
    #         total_reward += (1 - brax_state.done) * brax_state.reward
    #         return counter + 1, brax_state, total_reward, batch_step
    #
    #     init_val = (0, brax_state, jnp.zeros((batch_size, )), jnp.zeros((batch_size, )))
    #
    #     _counter, _brax_state, total_reward, batch_step = jax.lax.while_loop(
    #         cond_func, body_func, init_val
    #     )
    #     return total_reward, batch_step, state

    # @jit_method
    # def evaluate_multi(self, state, weights):
    #     batch_size = tree_leaves(weights)[0].shape[0]
    #     brax_state = self.jit_reset(jnp.tile(state.key, (batch_size, 1)))
    #
    #     def cond_func(val):
    #         counter, state, _total_reward, _ = val
    #         return (counter < self.cap_episode) & (~state.done.all())
    #
    #     def body_func(val):
    #         counter, brax_state, total_reward, batch_step = val
    #         action = self.batched_policy(weights, brax_state.obs)
    #         brax_state = self.jit_env_step(brax_state, action)
    #         batch_step += (1-brax_state.done)
    #         done = jnp.tile(brax_state.done[:, jnp.newaxis], (1, self.num_obj))
    #         reward = jnp.nan_to_num(brax_state.reward)
    #         total_reward += (1 - done) * reward
    #         return counter + 1, brax_state, total_reward, batch_step
    #
    #     init_val = (0, brax_state, jnp.zeros((batch_size, self.num_obj)), jnp.zeros((batch_size, )))
    #
    #     _counter, _brax_state, total_reward, batch_step = jax.lax.while_loop(
    #         cond_func, body_func, init_val
    #     )
    #     return total_reward, batch_step, state

    def visualize(
        self,
        state,
        key,
        weights,
        output_type: str = "HTML",
        respect_done=False,
        *args,
        **kwargs,
    ):
        assert output_type in [
            "HTML",
            "rgb_array",
        ], "output_type must be either HTML or rgb_array"

        env = brax_envs.get_environment(env_name=self.env_name, backend=self.backend)
        brax_state = jax.jit(env.reset)(key)
        jit_env_step = jit(env.step)
        trajectory = [brax_state.pipeline_state]
        episode_length = 1
        for _ in range(self.cap_episode):
            action = self.policy(weights, brax_state.obs)
            brax_state = jit_env_step(brax_state, action)
            trajectory.append(brax_state.pipeline_state)
            episode_length += 1 - brax_state.done

            if respect_done and brax_state.done:
                break

        if output_type == "HTML":
            return (
                html.render(env.sys.replace(dt=env.dt), trajectory, *args, **kwargs),
                state,
            )
        else:
            return (
                [
                    image.render_array(sys=self.env.sys, state=s, **kwargs)
                    for s in trajectory
                ],  # can use tqdm here
                state,
            )
