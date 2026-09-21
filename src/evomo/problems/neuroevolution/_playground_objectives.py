"""Task-specific maximization profiles for Playground 0.2 state-based tasks.

Metrics are transition values, never episode counters. Weighted metrics are
unscaled explicitly; native scalar rewards and their clipping are not reused
unless the source reward is itself a pure task-performance score.
"""

import math

import jax.numpy as jnp


def task_reward(previous, action, current):
    return current.reward


def action_efficiency(previous, action, current):
    """Positive command economy, not torque, work, or mechanical energy."""
    return 1.0 / (1.0 + jnp.mean(jnp.square(action)))


def metric(key, scale=1.0):
    if not math.isfinite(scale) or scale == 0:
        raise ValueError(f"Default objective requires a finite, nonzero reward scale for {key}; supply custom objectives")

    def objective(previous, action, current):
        return current.metrics[key] / scale

    return objective


def product(*functions):
    def objective(previous, action, current):
        return jnp.prod(jnp.stack([fn(previous, action, current) for fn in functions]))

    return objective


def default_objectives(name, env):
    """Return an ordered profile, rejecting unknown tasks instead of guessing."""

    def raw(key):
        return metric("reward/" + key)

    def weighted(key, prefix="reward/", scale_field="scales"):
        return metric(prefix + key, float(env._config.reward_config[scale_field][key]))

    # These native rewards contain no action penalties or mixed control terms.
    pure = {
        "AcrobotSwingup": "tip_target_proximity",
        "AcrobotSwingupSparse": "tip_target_success",
        "BallInCup": "ball_in_cup",
        "CheetahRun": "running_speed",
        "FingerSpin": "spin_success",
        "FingerTurnEasy": "target_contact",
        "FingerTurnHard": "target_contact",
        "PendulumSwingup": "upright_success",
        "ReacherEasy": "reach_success",
        "ReacherHard": "reach_success",
        "SwimmerSwimmer6": "target_proximity",
    }
    if name in pure:
        result = {pure[name]: task_reward}
    elif name in ("CartpoleBalance", "CartpoleSwingup"):
        result = {"pole_balance": product(raw("upright"), raw("small_velocity")), "cart_centering": raw("centered")}
    elif name in ("CartpoleBalanceSparse", "CartpoleSwingupSparse"):
        result = {"balance_success": product(raw("cart_in_bounds"), raw("angle_in_bounds"))}
    elif name == "PointMass":
        result = {"target_proximity": raw("near_target")}
    elif name == "FishSwim":
        result = {"target_proximity": raw("in_target"), "upright": raw("upright")}
    elif name in ("HopperHop", "HopperStand"):
        result = {"standing": raw("standing")}
        if name == "HopperHop":
            result["hopping"] = product(raw("standing"), raw("hopping"))
    elif name in ("WalkerStand", "WalkerWalk", "WalkerRun", "HumanoidStand", "HumanoidWalk", "HumanoidRun"):
        result = {"standing": raw("stand")}
        if name != "WalkerStand":
            result["stillness" if name == "HumanoidStand" else "locomotion"] = product(raw("stand"), raw("move"))
    elif name in ("Go1Getup", "SpotGetup"):
        result = {"recovery": product(weighted("orientation"), weighted("torso_height")), "posture": weighted("posture")}
    elif name in ("Go1Handstand", "Go1Footstand"):
        result = {"inverted_balance": product(weighted("orientation"), weighted("height"))}
    elif name == "H1InplaceGaitTracking":
        result = {"gait_tracking": weighted("feet_phase")}
        lin, ang = weighted("lin_vel"), weighted("ang_vel")
        result["body_stability"] = lambda p, a, n: 1 / (1 + lin(p, a, n) + ang(p, a, n))
    elif name == "ApolloJoystickFlatTerrain":
        # Apollo publishes a combined filtered-velocity score, not the two
        # separate tracking metrics used by other joystick environments.
        result = {"velocity_tracking": weighted("tracking"), "gait_tracking": weighted("feet_phase")}
    elif name in JOYSTICK_TASKS:
        result = {
            "linear_velocity_tracking": weighted("tracking_lin_vel"),
            "angular_velocity_tracking": weighted("tracking_ang_vel"),
        }
        if name in ("H1JoystickGaitTracking", "SpotJoystickGaitTracking"):
            result["gait_tracking"] = weighted("feet_phase")
    elif name in ("PandaPickCube", "PandaPickCubeOrientation", "PandaOpenCabinet"):
        result = {"object_target": metric("box_target")}
        key = "no_barrier_collision" if name == "PandaOpenCabinet" else "no_floor_collision"
        result["collision_avoidance"] = metric(key)
    elif name == "PandaPickCubeCartesian":
        result = {"object_target": raw("box_target"), "collision_avoidance": raw("no_floor_collision")}
    elif name == "PandaRobotiqPushCube":
        result = {
            "object_position": weighted("box_target", "", "reward_scales"),
            "object_orientation": weighted("box_orientation", "", "reward_scales"),
        }
    elif name == "AlohaHandOver":
        result = {"handover_target": weighted("handover_target", ""), "collision_avoidance": weighted("no_table_collision", "")}
    elif name == "AlohaSinglePegInsertion":
        result = {
            "peg_insertion": weighted("peg_insertion_reward", ""),
            "collision_avoidance": weighted("no_table_collision", ""),
        }
    elif name == "LeapCubeReorient":
        result = {"cube_orientation": weighted("orientation"), "cube_position": weighted("position")}
    elif name in ("LeapCubeRotateZAxis", "AeroCubeRotateZAxis"):
        # Upstream sets the linear-velocity weight to zero by default, so its
        # weighted metric has lost the signal. Read the physical state instead.
        result = {
            "cube_rotation": lambda p, a, n: env.get_cube_angvel(n.data)[2],
            "translation_stability": lambda p, a, n: 1 / (1 + jnp.sum(jnp.abs(env.get_cube_linvel(n.data)))),
        }
    else:
        raise ValueError(f"No default objective profile for {name!r}; supply an explicit objectives mapping")
    result["action_efficiency"] = action_efficiency
    return result


JOYSTICK_TASKS = {
    "BarkourJoystick",
    "BerkeleyHumanoidJoystickFlatTerrain",
    "BerkeleyHumanoidJoystickRoughTerrain",
    "G1JoystickFlatTerrain",
    "G1JoystickRoughTerrain",
    "Go1JoystickFlatTerrain",
    "Go1JoystickRoughTerrain",
    "H1JoystickGaitTracking",
    "Op3Joystick",
    "SpotFlatTerrainJoystick",
    "SpotJoystickGaitTracking",
    "T1JoystickFlatTerrain",
    "T1JoystickRoughTerrain",
}
