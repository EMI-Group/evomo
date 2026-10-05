"""Show MOEADFRRMAB's effective population and reward-window sizes."""

import torch

from evomo.algorithms import MOEADFRRMAB
from evomo.problems.numerical import DTLZ2
from evomo.workflows import UnifiedWorkflow


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    torch.manual_seed(42)
    problem = DTLZ2(d=12, m=3)
    algorithm = MOEADFRRMAB(
        pop_size=100,
        n_objs=problem.m,
        lb=torch.zeros(problem.d, device=device),
        ub=torch.ones(problem.d, device=device),
        T=20,
        delta=0.9,
        nr=2,
        window_size=None,
    )
    workflow = UnifiedWorkflow(algorithm, problem, device=device)
    workflow.init_step()
    for _ in range(3):
        workflow.step()

    assert torch.isfinite(workflow.algorithm.fit).all()
    print(f"Population shape: {tuple(workflow.algorithm.pop.shape)}")
    print(f"Reward window entries: {workflow.algorithm.W}")
    print(f"Fitness shape: {tuple(workflow.algorithm.fit.shape)}")


if __name__ == "__main__":
    main()
