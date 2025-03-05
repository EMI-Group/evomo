__all__ = [
    "mo_ant",
    "mo_half_cheetah",
    "mo_hopper",
    "mo_hopper_m3",
    "mo_humanoid",
    "mo_humanoidstandup",
    "mo_inverted_double_pendulum",
    "mo_pusher",
    "mo_reacher",
    "mo_swimmer",
    "mo_walker2d",
    "register_environment"
]

from brax.envs import register_environment

from evox_ext.problems.morobtrol import (
    mo_ant,
    mo_half_cheetah,
    mo_hopper,
    mo_hopper_m3,
    mo_humanoid,
    mo_humanoidstandup,
    mo_inverted_double_pendulum,
    mo_pusher,
    mo_reacher,
    mo_swimmer,
    mo_walker2d,
)

register_environment("mo_halfcheetah", mo_half_cheetah.MoHalfcheetah)
register_environment("mo_hopper_m2", mo_hopper.MoHopper)
register_environment("mo_hopper_m3", mo_hopper_m3.MoHopper)
register_environment("mo_swimmer", mo_swimmer.MoSwimmer)
register_environment("mo_humanoid", mo_humanoid.MoHumanoid)
register_environment(
    "mo_inverted_double_pendulum", mo_inverted_double_pendulum.MoInvertedDoublePendulum
)
register_environment("mo_walker2d", mo_walker2d.MoWalker2d)
register_environment("mo_humanoidstandup", mo_humanoidstandup.MoHumanoidStandup)
register_environment("mo_ant", mo_ant.MoAnt)
register_environment("mo_pusher", mo_pusher.MoPusher)
register_environment("mo_reacher", mo_reacher.MoReacher)
