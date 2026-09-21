"""Multiobjective Brax and Playground rollouts with a shared PyTorch evaluator."""

import copy
import weakref
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
from brax.io import html, image
from evox.core import Problem, use_state
from evox.problems.neuroevolution.utils import get_vmap_model_state_forward

from ._morobtrol_engines import available_playground_tasks, frames_to_html, make_environment


def to_jax_array(x: torch.Tensor) -> jax.Array:
    """Share a detached tensor via DLPack, falling back to CPU for CPU-only JAX."""
    x = x.detach().contiguous()
    if x.device.type != "cpu" and jax.default_backend() == "cpu":
        x = x.cpu()
    return jax.dlpack.from_dlpack(x)


def from_jax_array(x: jax.Array, device: Optional[torch.device] = None) -> torch.Tensor:
    return torch.utils.dlpack.from_dlpack(x).to(torch.get_default_device() if device is None else device)


# The registry keeps JAX callables outside torch.compile's traced state.
__brax_data__: Dict[int, tuple] = {}


def _call_policy(forward, state, obs):
    # PyTorch 2.11's compiled custom-op dispatcher excludes functorch keys.
    # This op starts a new, internal policy vmap, so enable just those keys
    # for this call and restore the caller's dispatch state on exit.
    keys = torch._C.DispatchKeySet(torch._C.DispatchKey.FuncTorchBatched)
    for name in ("FuncTorchDynamicLayerFrontMode", "FuncTorchDynamicLayerBackMode", "FuncTorchVmapMode"):
        keys = keys | torch._C.DispatchKeySet(getattr(torch._C.DispatchKey, name))
    with torch._C._ForceDispatchKeyGuard(
        torch._C._dispatch_tls_local_include_set(), torch._C._dispatch_tls_local_exclude_set() - keys
    ):
        return forward(state, obs)


def _normalize_obs(obs, stats, limits):
    stats = stats.to(torch.float64)
    count, m2, mean = stats[0], *stats[1:].chunk(2)
    variance = torch.clamp(m2 / count.clamp_min(1), min=limits[1], max=limits[2])
    # Missing features are neutral policy inputs, not artificial observations
    # to include in the running statistics. Preserve the policy input dtype.
    valid = torch.isfinite(obs)
    clean = torch.where(valid, obs.to(stats.dtype), torch.where(count > 0, mean, 0))
    normalized = torch.where(count > 0, (clean - mean) / variance.sqrt(), clean)
    return torch.clamp(normalized, min=-limits[0], max=limits[0]).to(obs.dtype)


def _update_obs_stats(stats, obs, active):
    """Merge finite active observations using float64 parallel Welford updates."""
    stats = stats.to(torch.float64)
    obs = obs.to(stats.dtype)
    count, m2, mean = stats[0], *stats[1:].chunk(2)
    # One shared count requires excluding the whole row if any feature is bad.
    active = active & torch.isfinite(obs).all(dim=-1)
    mask = active.unsqueeze(-1)
    n = active.sum().to(stats.dtype)
    batch_mean = torch.where(mask, obs, 0).sum(0) / n.clamp_min(1)
    batch_m2 = torch.where(mask, obs - batch_mean, 0).square().sum(0)
    total = count + n
    delta = batch_mean - mean
    new_mean = mean + delta * n / total.clamp_min(1)
    new_m2 = m2 + batch_m2 + delta.square() * count * n / total.clamp_min(1)
    return torch.cat((total.reshape(1), new_m2, new_mean))


@torch.inference_mode(False)
@torch.no_grad()
def _evaluate_brax_main(
    env_id: int,
    pop_size: int,
    rotate_key: bool,
    num_episodes: int,
    max_episode_length: int,
    key: torch.Tensor,
    model_state: List[torch.Tensor],
    num_obj: int,
    useless: bool,
    obs_param: torch.Tensor,
    obs_norm: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
    if not model_state or any(v.size(0) != pop_size for v in model_state):
        raise ValueError(f"All model state tensors must have leading population dimension {pop_size}")
    device = model_state[0].device
    reset, step, forward, state_keys = __brax_data__[env_id]
    # Isolate stateful policy buffers and avoid retaining autograd state.
    state = {k: v.clone() for k, v in zip(state_keys, model_state)}
    stats = obs_param.to(torch.float64).clone()
    key = to_jax_array(key)
    key, eval_key = jax.random.split(key) if rotate_key else (key, key)
    keys = jax.random.split(eval_key, num_episodes)
    keys = jnp.broadcast_to(keys, (pop_size, *keys.shape)).reshape(pop_size * num_episodes, -1)
    brax_state = reset(keys)
    done = jnp.zeros_like(brax_state.done, dtype=bool)
    reward_sum = jnp.zeros((pop_size * num_episodes, num_obj), dtype=brax_state.reward.dtype)
    for _ in range(max_episode_length):
        if bool(done.all()):
            break
        active = ~done
        raw_obs = from_jax_array(brax_state.obs, device)
        obs = raw_obs if useless else _normalize_obs(raw_obs, stats, obs_norm)
        state, action = _call_policy(forward, state, obs.reshape(pop_size, num_episodes, -1))
        if not useless:
            stats = _update_obs_stats(stats, raw_obs, from_jax_array(active, device))
        brax_state = step(brax_state, to_jax_array(action.reshape(pop_size * num_episodes, -1)))
        reward = jnp.nan_to_num(brax_state.reward)
        if reward.shape != reward_sum.shape:
            raise ValueError(f"Environment reward shape {reward.shape} does not match {reward_sum.shape}")
        # Include the terminal transition once, and never reactivate an episode.
        reward_sum += jnp.where(active[:, None], reward, 0)
        done |= brax_state.done.astype(bool)
    return (
        from_jax_array(key, device).clone(),
        [state[k].clone() for k in state_keys],
        from_jax_array(reward_sum, device).to(obs_norm.dtype).reshape(pop_size, num_episodes, num_obj).clone(),
        stats,
    )


@torch.library.custom_op("evomo::morobtrol_evaluate", mutates_args=())
def _evaluate_brax(
    env_id: int,
    pop_size: int,
    rotate_key: bool,
    num_episodes: int,
    max_episode_length: int,
    key: torch.Tensor,
    model_state: List[torch.Tensor],
    num_obj: int,
    useless: bool,
    obs_param: torch.Tensor,
    obs_norm: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
    return _evaluate_brax_main(
        env_id,
        pop_size,
        rotate_key,
        num_episodes,
        max_episode_length,
        key,
        model_state,
        num_obj,
        useless,
        obs_param,
        obs_norm,
    )


@_evaluate_brax.register_fake
def _fake_evaluate_brax(
    env_id,
    pop_size,
    rotate_key,
    num_episodes,
    max_episode_length,
    key,
    model_state,
    num_obj,
    useless,
    obs_param,
    obs_norm,
):
    return (
        torch.empty_like(key),
        [torch.empty_like(v) for v in model_state],
        obs_norm.new_empty(pop_size, num_episodes, num_obj),
        torch.empty_like(obs_param, dtype=torch.float64),
    )


@_evaluate_brax.register_vmap
def _evaluate_brax_vmap(
    info,
    in_dims,
    env_id,
    pop_size,
    rotate_key,
    num_episodes,
    max_episode_length,
    key,
    model_state,
    num_obj,
    useless,
    obs_param,
    obs_norm,
):
    if any(d is not None for i, d in enumerate(in_dims) if i != 6):
        raise ValueError("vmap over MoRobtrol state is unsupported; only policy parameters may be batched")
    dims = in_dims[6]
    if not any(d is not None for d in dims):
        raise ValueError("vmap requires batched policy parameters")
    flattened = [v if d is None else v.movedim(d, 0).flatten(0, 1) for v, d in zip(model_state, dims)]
    key, state, reward, stats = _evaluate_brax(
        env_id,
        pop_size,
        rotate_key,
        num_episodes,
        max_episode_length,
        key,
        flattened,
        num_obj,
        useless,
        obs_param,
        obs_norm,
    )
    state = [v.unflatten(0, (info.batch_size, -1)) for v in state]
    reward = reward.unflatten(0, (info.batch_size, -1))
    return (key, state, reward, stats), (None, [0] * len(state), 0, None)


class MoRobtrol(Problem):
    """Evaluate PyTorch policies in multiobjective Brax or Playground environments.

    ``engine='brax'`` preserves existing behavior. ``engine='playground'`` supports
    four legacy mo_* ports and native Playground registry names, and requires the
    ``playground`` extra. Its backend must be None or 'mjx'. Native tasks use
    task-specific profiles with two to four maximized objectives. Inspect
    ``objective_names`` and ``num_obj`` instead of assuming two objectives.
    ``objectives`` maps names to scalar JAX functions (previous, action, next).
    ``env_config`` contains native config overrides. Dictionary observations use
    ``observation_key='state'`` by default; images/other selected arrays flatten.
    Both engines share the stepwise evaluator, including normalization and RNG.
    Different physics backends need not produce interchangeable fitness values.

    Fitness has shape ``(pop_size, num_obj)`` and includes terminal rewards.
    Individuals share the same episode seeds. ``rotate_key=False`` makes seeds
    repeat; with adaptive normalization enabled, evolving statistics can still
    change fitness between calls.

    ``num_obj`` and ``observation_shape`` are inferred from the environment when
    omitted. ``useless=True`` retains the historical default of no observation
    normalization. Set it to False to normalize policy inputs using running
    count/M2/mean statistics. ``obs_norm`` is ``[clip, min_variance, max_variance]``.
    Before any observations have been collected, normalization uses unit scale.
    Statistics accumulate in float64 and exclude rows with nonfinite features.
    Nonfinite policy-input features normalize to zero; input dtype is preserved.

    For an outer vmap over policy parameters, set ``pop_size`` to outer batch
    size times inner population size. RNG and normalization statistics are shared
    over this combined population. Mapping independent problem states or nested
    HPO repeats is unsupported; use ``num_episodes`` for repeated evaluations.
    """

    def __init__(
        self,
        policy: nn.Module,
        env_name: str,
        max_episode_length: int,
        num_episodes: int,
        seed: int = None,
        pop_size: int | None = None,
        rotate_key: bool = True,
        reduce_fn: Callable = torch.mean,
        backend: str | None = None,
        device: torch.device | None = None,
        num_obj: int | None = None,
        observation_shape: int | None = None,
        obs_norm: torch.Tensor = None,
        useless: bool = True,
        *,
        engine: str = "brax",
        objectives: Optional[Dict[str, Callable]] = None,
        env_config: Optional[dict] = None,
        observation_key: Optional[str] = None,
    ):
        super().__init__()
        device = torch.get_default_device() if device is None else device
        pop_size = 1 if pop_size is None else pop_size
        for name, value in (("pop_size", pop_size), ("num_episodes", num_episodes)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(max_episode_length, int) or isinstance(max_episode_length, bool) or max_episode_length < 0:
            raise ValueError("max_episode_length must be a nonnegative integer")
        env = make_environment(
            env_name, engine, backend, objectives=objectives, env_config=env_config, observation_key=observation_key
        )
        self.engine = engine
        self._environment = env
        self.objective_names = getattr(env, "objective_names", None)
        expected_obj = getattr(env, "num_obj", 1)
        if num_obj is not None and num_obj != expected_obj:
            raise ValueError(f"num_obj must match environment: expected {expected_obj}, got {num_obj}")
        if observation_shape not in (None, 0, env.observation_size):
            raise ValueError(f"observation_shape must match environment: expected {env.observation_size}")
        self.num_obj = expected_obj
        self.observation_shape = env.observation_size
        self.pop_size, self.num_episodes = pop_size, num_episodes
        self.max_episode_length = max_episode_length
        self.rotate_key, self.reduce_fn, self.useless = rotate_key, reduce_fn, useless
        limits = torch.as_tensor([5.0, 1e-6, 1e6] if obs_norm is None else obs_norm, device=device).detach().clone()
        if limits.shape != (3,) or not bool(torch.isfinite(limits).all()) or not bool((limits > 0).all()):
            raise ValueError("obs_norm must contain three finite positive values")
        if bool(limits[1] > limits[2]):
            raise ValueError("obs_norm minimum variance must not exceed maximum variance")
        self.register_buffer("obs_norm", limits.to(torch.get_default_dtype()))
        self.register_buffer("obs_param", torch.zeros(1 + 2 * self.observation_shape, device=device, dtype=torch.float64))
        seed = torch.randint(0, 2**31, ()).item() if seed is None else seed
        self.register_buffer("key", from_jax_array(jax.random.PRNGKey(seed), device).clone())

        copied_policy = copy.deepcopy(policy).to(device)
        self.state_keys = list(copied_policy.state_dict())
        if not self.state_keys:
            raise ValueError("policy must have at least one parameter or buffer")
        # Register defaults so checkpoint loading and .to() also cover policy buffers.
        self._initial_state = nn.Module()
        for i, value in enumerate(copied_policy.state_dict().values()):
            self._initial_state.register_buffer(f"s{i}", value.detach().clone())
        _, self.vmap_state_forward = get_vmap_model_state_forward(
            model=copied_policy,
            pop_size=pop_size,
            in_dims=(0, 0),
            device=device,
        )
        self.state_forward = use_state(copied_policy)
        self.brax_reset, self.brax_step = jax.jit(env.reset), jax.jit(env.step)
        self.vmap_brax_reset, self.vmap_brax_step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
        self.env_sys = env.sys if engine == "brax" else None
        self._id_ = id(self)
        __brax_data__[self._id_] = self.vmap_brax_reset, self.vmap_brax_step, self.vmap_state_forward, self.state_keys
        weakref.finalize(self, __brax_data__.pop, self._id_, None)

    @property
    def device(self):
        return self.key.device

    @staticmethod
    def available_playground_tasks():
        """Return legacy ports plus task names registered in installed Playground."""
        return available_playground_tasks()

    @staticmethod
    def playground_task_info(env_name, *, env_config=None, observation_key=None, objectives=None):
        """Load task metadata to size a policy before constructing the problem.

        Native robot assets may be downloaded by Playground on first use.
        Pass the same configuration options when constructing MoRobtrol.
        """
        env = make_environment(
            env_name, "playground", "mjx", objectives=objectives, env_config=env_config, observation_key=observation_key
        )
        return dict(
            observation_size=env.observation_size,
            action_size=env.action_size,
            num_obj=env.num_obj,
            objective_names=getattr(env, "objective_names", None),
        )

    @property
    def init_state(self):
        return {k: getattr(self._initial_state, f"s{i}") for i, k in enumerate(self.state_keys)}

    @property
    def vmap_init_state(self):
        return {k: v.unsqueeze(0).expand(self.pop_size, *v.shape) for k, v in self.init_state.items()}

    def evaluate(self, pop_params: Dict[str, nn.Parameter]) -> torch.Tensor:
        """Return episode-reduced objective rewards for batched policy parameters."""
        state = {**self.vmap_init_state, **pop_params}
        key, _, rewards, stats = _evaluate_brax(
            self._id_,
            self.pop_size,
            self.rotate_key,
            self.num_episodes,
            self.max_episode_length,
            self.key,
            [state[k] for k in self.state_keys],
            self.num_obj,
            self.useless,
            self.obs_param,
            self.obs_norm,
        )
        self.key, self.obs_param = key, stats
        return self.reduce_fn(rewards, dim=1)

    @torch.compiler.disable
    @torch.no_grad()
    def _evaluate_brax_record(self, model_state, seed=0):
        # Recording uses its own seed and frozen statistics, leaving evaluation state untouched.
        state = self.brax_reset(jax.random.PRNGKey(seed))
        trajectory = [state.pipeline_state if self.engine == "brax" else state]
        reward = jnp.zeros((self.num_obj,))
        model_state = {k: v.clone() for k, v in model_state.items()}
        for _ in range(self.max_episode_length):
            if bool(state.done):
                break
            obs = from_jax_array(state.obs, self.device)
            if not self.useless:
                obs = _normalize_obs(obs, self.obs_param, self.obs_norm)
            model_state, action = self.state_forward(model_state, obs)
            state = self.brax_step(state, to_jax_array(action))
            reward += jnp.nan_to_num(state.reward)
            trajectory.append(state.pipeline_state if self.engine == "brax" else state)
        return model_state, from_jax_array(reward, self.device), trajectory

    def visualize(
        self, weights: Dict[str, nn.Parameter], seed: int = 0, output_type: str = "HTML", *args, **kwargs
    ) -> str | List[np.ndarray]:
        """Render a seeded rollout as HTML or a list of RGB frames without changing evaluation state."""
        if output_type not in ("HTML", "rgb_array"):
            raise ValueError("output_type must be 'HTML' or 'rgb_array'")
        _, _, trajectory = self._evaluate_brax_record({**self.init_state, **weights}, seed=seed)
        if self.engine == "playground":
            frames = self._environment.render(trajectory, *args, **kwargs)
            return frames_to_html(frames) if output_type == "HTML" else frames
        if output_type == "HTML":
            return html.render(self.env_sys, trajectory, *args, **kwargs)
        return image.render_array(self.env_sys, trajectory, *args, **kwargs)
