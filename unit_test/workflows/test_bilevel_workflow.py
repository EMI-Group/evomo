"""BilevelWorkflow integration with existing algorithms and compiled execution."""

from functools import partial

import pytest
import torch

from evomo.algorithms import NSGA2, NSGA3
from evomo.operators.selection import non_dominate as nd_ops
from evomo.problems.bilevel import DS1D, TP1, TP2
from evomo.workflows import BilevelWorkflow

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))]


def make_workflow(problem=None, upper=NSGA2, lower=NSGA2, **kwargs):
    if problem is None:
        problem = TP2(d_l=2, device="cpu", dtype=torch.float64)
    options = dict(lower_generations=1, lower_restarts=2, archive_size=8) | kwargs
    return BilevelWorkflow(problem, partial(upper, pop_size=4), partial(lower, pop_size=4), **options)


def assert_archive(workflow):
    finite = torch.isfinite(workflow.fit).all(-1)
    evaluated = workflow.problem.evaluate_all(workflow.pop[finite])
    torch.testing.assert_close(workflow.fit[finite], evaluated.upper_fitness)
    torch.testing.assert_close(workflow.lower_fit[finite], evaluated.lower_fitness)
    physical_cv = torch.cat((evaluated.upper_cv, evaluated.lower_cv), -1)
    torch.testing.assert_close(workflow.cv[finite, 1:], physical_cv)
    assert torch.isfinite(workflow.pop).all()
    assert not torch.isnan(workflow.cv).any()
    assert workflow.pop.shape[-1] == workflow.problem.d
    assert workflow.upper_algorithm.pop.shape[-1] == workflow.problem.d_u + workflow.problem.m_l
    for value in (workflow.pop, workflow.fit, workflow.lower_fit, workflow.cv):
        assert value.device == workflow.problem.device and value.dtype == workflow.problem.dtype


@pytest.mark.parametrize("upper,lower", [(NSGA2, NSGA2), (NSGA2, NSGA3), (NSGA3, NSGA2), (NSGA3, NSGA3)])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bilevel_workflow_algorithms(upper, lower, device, dtype):
    workflow = make_workflow(TP2(d_l=2, device=device, dtype=dtype), upper, lower)
    assert int(workflow.lower_evaluations) == 4  # Template initialization is counted.
    workflow.init_step()
    workflow.step()
    assert_archive(workflow)
    assert int(workflow.generation) == 1
    assert int(workflow.upper_evaluations) == 2 * 4 * 2 * 4
    assert int(workflow.lower_evaluations) == 4 + 2 * 4 * 2 * 4 * 3
    assert workflow.solution_mask.any()


@pytest.mark.parametrize("cls", [TP1, DS1D])
@pytest.mark.parametrize("device", DEVICES)
def test_bilevel_workflow_problems(cls, device):
    problem = cls(device=device, dtype=torch.float64)
    workflow = make_workflow(problem)
    workflow.init_step()
    workflow.step()
    assert_archive(workflow)
    assert workflow.cv.shape[-1] == 1 + problem.n_iq_u + problem.n_eq_u + problem.n_iq_l + problem.n_eq_l


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("constrained", [False, True])
def test_bilevel_workflow_compiled(device, constrained, monkeypatch):
    torch.compiler.reset()
    if device == "cpu":
        # These custom sorting operators separately compile their loops.
        # Graph capture must not depend on a Windows C++ toolchain.
        for name in ("_partial_rank_compile", "_iterative_get_ranks_compile", "_vmap_iterative_get_ranks_compile"):
            monkeypatch.setattr(
                nd_ops, name, torch.compile(getattr(nd_ops, name)._torchdynamo_orig_callable, backend="eager", fullgraph=True)
            )
    factory = TP1 if constrained else partial(TP2, d_l=2)
    eager = make_workflow(factory(device=device, dtype=torch.float64))
    captured = make_workflow(factory(device=device, dtype=torch.float64))
    eager.init_step()
    captured.init_step()
    captured.load_state_dict(eager.state_dict())
    torch.manual_seed(23)
    eager.step()
    torch.manual_seed(23)
    torch.compile(captured.step, backend="eager", fullgraph=True)()
    for key, value in eager.state_dict().items():
        torch.testing.assert_close(value, captured.state_dict()[key])
    assert_archive(captured)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("cls", [NSGA2, NSGA3])
def test_bilevel_workflow_cuda_inductor(cls):
    torch.compiler.reset()
    workflow = make_workflow(TP2(d_l=2, device="cuda", dtype=torch.float64), upper=cls, lower=cls, lower_accuracy_tol=0.1)
    workflow.init_step()
    compiled = torch.compile(workflow.step, fullgraph=True)
    compiled()
    compiled()
    assert_archive(workflow)
    assert int(workflow.generation) == 2
    assert int(workflow.lower_evaluations) == 4 + 3 * 4 * 2 * 4 * 3
