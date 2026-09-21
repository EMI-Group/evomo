"""Native Playground tasks with explicit, configurable vector objectives."""

import copy
import math
from collections.abc import Mapping

import jax
import jax.numpy as jnp
from flax import struct
from mujoco_playground import registry

from ._playground_objectives import default_objectives


@struct.dataclass
class NativeState:
    # Keep the native observation/reward intact: some tasks read them in step().
    native: object
    obs: jax.Array
    reward: jax.Array
    done: jax.Array


class NativePlayground:
    """Adapt any registered state-observation task without changing its dynamics.

    Custom objectives map names to JAX-compatible functions of
    (previous_native_state, action, next_native_state). All are maximized and
    accumulated over active transitions, including the terminal transition.
    """

    def __init__(self, name, *, objectives=None, env_config=None, observation_key=None):
        if name not in registry.ALL_ENVS:
            raise ValueError(
                f"Playground task {name!r} is not implemented/registered; available: {', '.join(registry.ALL_ENVS)}"
            )
        overrides = copy.deepcopy(dict(env_config or {}))
        if overrides.get("impl", "jax") != "jax":
            raise ValueError("MoRobtrol's MJX backend requires env_config['impl']='jax', not Warp")
        config = registry.get_default_config(name)
        if "impl" in config:
            config.impl = "jax"
        self.env = registry.load(name, config=config, config_overrides=overrides)
        if objectives is None:
            objectives = default_objectives(name, self.env)
        if not isinstance(objectives, Mapping) or len(objectives) < 2:
            raise ValueError("objectives must map at least two names to scalar JAX objective functions")
        if any(not isinstance(k, str) or not k or not callable(v) for k, v in objectives.items()):
            raise ValueError("objective names must be nonempty strings and values must be callable")
        self.objective_names = tuple(objectives)
        self._objectives = tuple(objectives.values())
        self.num_obj = len(self._objectives)
        self.action_size = self.env.action_size
        shapes = self.env.observation_size
        if isinstance(shapes, Mapping):
            if observation_key is None:
                if "state" not in shapes:
                    raise ValueError(f"Dictionary observation requires observation_key; choose from {tuple(shapes)}")
                observation_key = "state"
            if observation_key not in shapes:
                raise ValueError(f"Unknown observation_key {observation_key!r}; choose from {tuple(shapes)}")
            shape = shapes[observation_key]
        else:
            if observation_key is not None:
                raise ValueError("observation_key is only valid for dictionary observations")
            shape = shapes
        self.observation_key = observation_key
        self.observation_size = int(shape) if isinstance(shape, int) else math.prod(shape)
        if self.observation_size < 1:
            raise ValueError("Selected observation must not be empty")

    def _obs(self, state):
        obs = state.obs if self.observation_key is None else state.obs[self.observation_key]
        return jnp.asarray(obs).reshape(-1)

    def reset(self, key):
        native = self.env.reset(key)
        return NativeState(native, self._obs(native), jnp.zeros(self.num_obj), native.done)

    def step(self, state, action):
        # Native tasks may mutate nested info/metrics dictionaries while tracing.
        # Copy containers so custom objectives can reliably inspect pre-step data.
        previous = state.native
        current = self.env.step(jax.tree.map(lambda leaf: leaf, previous), action)
        values = [jnp.asarray(fn(previous, action, current)) for fn in self._objectives]
        if any(value.shape != () for value in values):
            raise ValueError("Each objective function must return a scalar")
        return NativeState(current, self._obs(current), jnp.stack(values), current.done)

    def render(self, trajectory, *args, **kwargs):
        return self.env.render([state.native for state in trajectory], *args, **kwargs)
