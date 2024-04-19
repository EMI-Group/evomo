from brax.envs import register_environment
from problems.mo_brax import (
    mo_half_cheetah,
    mo_swimmer,
    mo_hopper,
    mo_hopper_m3,
    mo_humanoid,
    mo_inverted_double_pendulum,
    mo_walker2d,
    mo_humanoidstandup,
    mo_ant,
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
