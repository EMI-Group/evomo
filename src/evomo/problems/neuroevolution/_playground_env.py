"""Independent Playground/MJX ports of four existing MoRobtrol tasks.

Uses the installed Brax Apache-2.0 XML assets, not its physics pipeline.
Default task equations follow brax.envs and EvoMO's multiobjective extensions.
Used by MoRobtrol with engine="playground"; rollout evaluation is shared with Brax.
"""

from importlib.resources import files

import jax
import jax.numpy as jp
import mujoco
from ml_collections import ConfigDict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from ._morobtrol_engines import PLAYGROUND_TASKS


class MoPlayground(mjx_env.MjxEnv):
    def __init__(self, name):
        if name not in PLAYGROUND_TASKS:
            raise ValueError(f"Playground port does not implement {name}")
        asset, frames, self.num_obj = PLAYGROUND_TASKS[name]
        self.name = name
        self._xml_path = str(files("brax") / "envs" / "assets" / f"{asset}.xml")
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mjx_model = mjx.put_model(self._mj_model)
        dt = self._mj_model.opt.timestep
        super().__init__(ConfigDict(dict(ctrl_dt=dt * frames, sim_dt=dt)))
        self._init_q = jp.asarray(self._mj_model.qpos0)
        # Brax custom init_qpos takes precedence over XML joint defaults.
        for i in range(self._mj_model.nnumeric):
            if mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_NUMERIC, i) == "init_qpos":
                adr, size = self._mj_model.numeric_adr[i], self._mj_model.numeric_size[i]
                self._init_q = jp.asarray(self._mj_model.numeric_data[adr : adr + size])
        self._init_z = self._mj_model.body_pos[1, 2]

    @property
    def xml_path(self):
        return self._xml_path

    @property
    def mj_model(self):
        return self._mj_model

    @property
    def mjx_model(self):
        return self._mjx_model

    @property
    def action_size(self):
        return self._mj_model.nu

    def _obs(self, data):
        if self.name == "mo_swimmer":
            return jp.concatenate((data.qpos[2:], data.qvel))
        if self.name.startswith("mo_hopper"):
            q = data.qpos.at[1].set(data.xpos[1, 2])
            return jp.concatenate((q[1:], jp.clip(data.qvel, -10, 10)))
        return jp.concatenate((data.qpos[1:], data.qvel))

    def reset(self, rng):
        _, key_q, key_v = jax.random.split(rng, 3)
        scale = 0.005 if self.name.startswith("mo_hopper") else 0.1
        q = self._init_q + jax.random.uniform(key_q, (self.mj_model.nq,), minval=-scale, maxval=scale)
        if self.name == "mo_halfcheetah":
            v = scale * jax.random.normal(key_v, (self.mj_model.nv,))
        else:
            v = jax.random.uniform(key_v, (self.mj_model.nv,), minval=-scale, maxval=scale)
        data = mjx.make_data(self.mj_model).replace(qpos=q, qvel=v)
        data = mjx.forward(self.mjx_model, data)
        return mjx_env.State(
            data=data, obs=self._obs(data), reward=jp.zeros(self.num_obj), done=jp.array(0.0), metrics={}, info={}
        )

    def step(self, state, action):
        data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
        velocity = (data.xpos[1, 0] - state.data.xpos[1, 0]) / self.dt
        if self.name == "mo_swimmer":
            velocity = (data.qpos[0] - state.data.qpos[0]) / self.dt
        energy = -jp.sum(jp.square(action))
        done = jp.array(0.0)
        if self.name.startswith("mo_hopper"):
            z, angle = data.xpos[1, 2], data.qpos[2]
            state_vec = jp.concatenate((data.qpos[2:], data.qvel))
            healthy = jp.all((state_vec > -100) & (state_vec < 100)) & (z > 0.7) & (z < jp.inf) & (jp.abs(angle) < 0.2)
            done = 1.0 - healthy.astype(jp.float32)
            height = 10 * (z - self._init_z)
            if self.num_obj == 3:
                reward = jp.array([velocity, height, energy]) + 1.0
            else:
                reward = jp.array([velocity, height]) + 0.001 * energy + 1.0
        else:
            weight = 1e-4 if self.name == "mo_swimmer" else 0.1
            reward = jp.array([velocity, weight * energy])
        return state.replace(data=data, obs=self._obs(data), reward=reward, done=done)
