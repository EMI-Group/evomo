"""Keep constraint violations when solving DOC1 with NSGA-II."""

import torch

from evomo.algorithms import NSGA2
from evomo.operators.selection import non_dominate_rank
from evomo.problems.constrained import DOC1
from evomo.workflows import UnifiedWorkflow


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    torch.manual_seed(42)

    problem = DOC1()
    algorithm = NSGA2(pop_size=100, n_objs=problem.m, lb=problem.lb, ub=problem.ub, device=device)
    workflow = UnifiedWorkflow(algorithm, problem, device=device)
    workflow.init_step()
    for _ in range(50):
        workflow.step()

    fitness = workflow.algorithm.fit
    cv = workflow.algorithm.cv
    feasible = (cv.sum(dim=1) <= 0) & torch.isfinite(fitness).all(dim=1)
    feasible_fitness = fitness[feasible]
    print(f"Fitness: {tuple(fitness.shape)}, CV: {tuple(cv.shape)}")
    print(f"Finite feasible solutions: {feasible.sum().item()}/{feasible.numel()}")
    if feasible_fitness.shape[0] > 0:
        print(feasible_fitness[non_dominate_rank(feasible_fitness) == 0])
    else:
        print("No feasible solution in the current population; do not report a feasible-front metric.")


if __name__ == "__main__":
    main()
