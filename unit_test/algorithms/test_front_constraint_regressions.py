"""Keep main-branch front selection fixes intact when adding constraint handling."""

import pytest
import torch
from evox.core import Problem

from evomo.algorithms import SIBEA, GrEA, VaEA, grea
from evomo.workflows import UnifiedWorkflow

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class FeasibleObjectives(Problem):
    def __init__(self, constrained):
        super().__init__()
        self.constrained = constrained

    def evaluate(self, pop):
        fit = torch.stack([pop.square().sum(1), (1 - pop).square().sum(1)], dim=1)
        return (fit, pop.new_zeros((pop.shape[0], 2))) if self.constrained else fit


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("algorithm_cls", [GrEA, VaEA, SIBEA])
def test_zero_violation_preserves_objective_only_selection(algorithm_cls, device):
    results = []
    for constrained in (False, True):
        torch.manual_seed(42)
        lb = torch.zeros(6, device=device)
        algorithm = algorithm_cls(pop_size=12, n_objs=2, lb=lb, ub=lb + 1)
        workflow = UnifiedWorkflow(algorithm, FeasibleObjectives(constrained), device=device)
        workflow.init_step()
        for _ in range(2):
            workflow.step()
            assert algorithm.fit.shape == (12, 2)
        if constrained:
            assert algorithm.cv.shape == (12, 2)
            assert algorithm.cv.device == algorithm.pop.device
            assert not algorithm.cv.any()
        else:
            assert algorithm.cv is None
        results.append((algorithm.pop.clone(), algorithm.fit.clone()))
    for objective_only, feasible in zip(*results):
        torch.testing.assert_close(objective_only, feasible, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_grea_mating_prioritizes_violation_before_native_grid_rules(device, monkeypatch):
    lb = torch.zeros(2, device=device)
    algorithm = GrEA(pop_size=4, n_objs=2, lb=lb, ub=lb + 1)
    algorithm.fit = torch.tensor([[0, 0], [1, 1], [0, 0], [2, 2]], dtype=torch.float32, device=device)
    algorithm.cv = torch.tensor([[2, -1], [1, -1], [-1, -1], [0, -1]], dtype=torch.float32, device=device)
    pairs = iter([torch.tensor([0, 1, 2, 3], device=device), torch.tensor([1, 0, 3, 2], device=device)])
    monkeypatch.setattr(grea.torch, "randint", lambda *args, **kwargs: next(pairs))
    torch.testing.assert_close(algorithm._mating_selection(), torch.tensor([1, 1, 2, 2], device=device))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("violation", [0.0, 1.0])
def test_vaea_worse_elimination_still_operates_at_equal_violation(device, violation):
    fit = torch.tensor(
        [[0, 1], [0.2, 0.85], [0.3, 0.75], [0.49, 0.515], [0.51, 0.50], [0.7, 0.32], [0.8, 0.21], [1, 0]],
        device=device,
    )
    lb = torch.zeros(3, device=device)
    algorithm = VaEA(pop_size=4, n_objs=2, lb=lb, ub=lb + 1)
    algorithm.fit = fit[:4].clone()
    algorithm.cv = torch.full((4, 1), violation, device=device)
    algorithm.evaluate = lambda pop: (fit[4:], torch.full((4, 1), violation, device=device))
    algorithm.step()
    assert (algorithm.fit == fit[3]).all(1).any()
    assert not (algorithm.fit == fit[4]).all(1).any()
    assert algorithm.cv.shape == (4, 1)
    assert (algorithm.cv == violation).all()


@pytest.mark.parametrize("device", DEVICES)
def test_vaea_worse_elimination_preserves_better_constraint_fronts(device, monkeypatch):
    # A worse-converged feasible incumbent must survive a nearby, better-converged infeasible candidate.
    fit = torch.tensor([[3, 3], [2, 2], [0, 0], [0.4, 0.5], [1, 1], [0.5, 1], [1, 0.5], [1, 0]], device=device)
    cv = torch.ones(8, 1, device=device)
    cv[0] = 0
    angles = torch.ones(8, 8, device=device)
    angles[0, 2:] = 0.001
    angles[2:, 0] = 0.001
    angles.fill_diagonal_(0)
    lb = torch.zeros(3, device=device)
    algorithm = VaEA(pop_size=4, n_objs=2, lb=lb, ub=lb + 1)
    algorithm.fit = fit[:4].clone()
    algorithm.cv = cv[:4].clone()
    algorithm.evaluate = lambda pop: (fit[4:], cv[4:])
    monkeypatch.setattr(algorithm, "_cosine_distance_matrix", lambda _: angles)
    algorithm.step()
    assert algorithm.fit.shape == (4, 2)
    assert (algorithm.fit == fit[0]).all(1).any()
    assert (algorithm.cv == 0).sum() == 1
