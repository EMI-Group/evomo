"""A batched, two-objective problem integrated with an EvoMO workflow."""

import torch
from evox.core import Problem

from evomo.algorithms import NSGA2
from evomo.workflows import UnifiedWorkflow


class TwoCenters(Problem):
    def evaluate(self, population: torch.Tensor) -> torch.Tensor:
        objective_a = population.square().sum(dim=1)
        objective_b = (population - 1).square().sum(dim=1)
        return torch.stack((objective_a, objective_b), dim=1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    torch.manual_seed(42)
    algorithm = NSGA2(pop_size=100, n_objs=2, lb=torch.zeros(10), ub=torch.ones(10), device=device)
    workflow = UnifiedWorkflow(algorithm, TwoCenters(), device=device)
    workflow.init_step()
    for _ in range(30):
        workflow.step()
    print(workflow.algorithm.fit)


if __name__ == "__main__":
    main()
