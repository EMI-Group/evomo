"""Optimize DTLZ2 with NSGA-II and report the current population's IGD."""

import torch

from evomo.algorithms import NSGA2
from evomo.metrics import igd
from evomo.operators.selection import non_dominate_rank
from evomo.problems.numerical import DTLZ2
from evomo.workflows import UnifiedWorkflow


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    torch.manual_seed(42)

    problem = DTLZ2(d=12, m=3)
    algorithm = NSGA2(
        pop_size=100,
        n_objs=problem.m,
        lb=torch.zeros(problem.d, device=device),
        ub=torch.ones(problem.d, device=device),
        device=device,
    )
    workflow = UnifiedWorkflow(algorithm, problem, device=device)
    workflow.init_step()
    for _ in range(50):
        workflow.step()

    fitness = workflow.algorithm.fit
    finite = torch.isfinite(fitness).all(dim=1)
    population = workflow.algorithm.pop[finite]
    fitness = fitness[finite]
    if fitness.shape[0] == 0:
        raise RuntimeError("No finite objective values; check the problem evaluation")
    front = non_dominate_rank(fitness) == 0
    pareto_solutions = population[front]
    pareto_fitness = fitness[front]
    reference_front = problem.pf().to(device=fitness.device, dtype=fitness.dtype)

    print(f"Device: {device}")
    print(f"Population: {tuple(workflow.algorithm.pop.shape)}")
    print(f"Non-dominated solutions: {pareto_solutions.shape[0]}")
    print(f"IGD: {igd(pareto_fitness, reference_front).item():.6f}")


if __name__ == "__main__":
    main()
