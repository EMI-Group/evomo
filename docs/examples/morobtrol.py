"""Search a small MoSwimmer policy with the optional Brax engine."""

import jax
import torch
from brax import envs
from evox.utils import ParamsAndVector

from evomo.algorithms import NSGA2
from evomo.problems.neuroevolution import MoRobtrol
from evomo.workflows import UnifiedWorkflow


def main():
    jax_gpu = any(device.platform == "gpu" for device in jax.devices())
    device = "cuda" if torch.cuda.is_available() and jax_gpu else "cpu"
    torch.set_default_device(device)
    torch.manual_seed(42)

    env = envs.get_environment("mo_swimmer")
    policy = torch.nn.Sequential(
        torch.nn.Linear(env.observation_size, 16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, env.action_size),
        torch.nn.Tanh(),
    ).to(device)
    adapter = ParamsAndVector(dummy_model=policy)
    center = adapter.to_vector(dict(policy.named_parameters()))
    population_size = 20
    problem = MoRobtrol(
        policy=policy,
        env_name="mo_swimmer",
        max_episode_length=30,
        num_episodes=2,
        seed=42,
        pop_size=population_size,
        rotate_key=False,
        device=device,
    )
    algorithm = NSGA2(
        pop_size=population_size,
        n_objs=problem.num_obj,
        lb=torch.full_like(center, -1),
        ub=torch.full_like(center, 1),
        device=device,
    )
    workflow = UnifiedWorkflow(algorithm, problem, opt_direction="max", solution_transform=adapter, device=device)
    workflow.init_step()
    for _ in range(3):
        workflow.step()
    print("Objective rewards:")
    print(-workflow.algorithm.fit)


if __name__ == "__main__":
    main()
