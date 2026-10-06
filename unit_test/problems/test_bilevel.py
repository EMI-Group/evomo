"""Bilevel problem evaluation, reference fronts and workflow integration."""

import pytest
import torch

from evomo.algorithms import NSGA2
from evomo.problems import bilevel
from evomo.workflows import UnifiedWorkflow

PROBLEMS = [getattr(bilevel, prefix + str(i)) for prefix, count in (("TP", 4), ("DS", 5)) for i in range(1, count + 1)]
DECEPTIVE = [bilevel.DS1D, bilevel.DS2D, bilevel.DS3D]
PROBLEMS += DECEPTIVE


def fitness(result):
    return result[0] if isinstance(result, tuple) else result


@pytest.mark.parametrize("cls", PROBLEMS)
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bilevel_problems(cls, device, dtype):
    problem = cls(device=device, dtype=dtype)
    assert problem.m_u >= 2 and problem.m_l >= 2
    z = problem.lb + (problem.ub - problem.lb) * torch.rand(2, 3, problem.d, device=device, dtype=dtype)
    before = z.clone()
    result = problem.evaluate_all(z)
    assert result.upper_fitness.shape == (2, 3, problem.m_u)
    assert result.lower_fitness.shape == (2, 3, problem.m_l)
    assert result.upper_cv.shape == (2, 3, problem.n_iq_u + problem.n_eq_u)
    assert result.lower_cv.shape == (2, 3, problem.n_iq_l + problem.n_eq_l)
    for value in result:
        assert value.dtype == dtype and value.device.type == device
        assert torch.isfinite(value).all()
    torch.testing.assert_close(z, before, rtol=0, atol=0)
    torch.testing.assert_close(fitness(problem.evaluate(z)), result.upper_fitness)
    # Bind one xu to a whole follower population without copying or Python loops.
    xu, xl = problem.split(z)
    batch_lower = fitness(problem.evaluate_lower(xu[:, :1], xl))
    for i in range(2):
        for j in range(3):
            torch.testing.assert_close(batch_lower[i, j], fitness(problem.evaluate_lower(xu[i, 0], xl[i, j])))
    for batch in ((), (0,), (2, 0)):
        empty_or_single = z.new_empty((*batch, problem.d))
        if not batch:
            empty_or_single = z[0, 0]
        result = problem.evaluate_all(empty_or_single)
        assert result.upper_fitness.shape == (*batch, problem.m_u)
        assert result.lower_cv.shape == (*batch, problem.n_iq_l + problem.n_eq_l)


@pytest.mark.parametrize("cls", [bilevel.TP1, bilevel.TP2, bilevel.DS1, bilevel.DS3, bilevel.DS5] + DECEPTIVE)
def test_bilevel_problems_compiled(cls):
    problem = cls(dtype=torch.float64)
    z = problem.lb + (problem.ub - problem.lb) * torch.rand(8, problem.d, dtype=torch.float64)
    expected = problem.evaluate_all(z)
    # Eager backend verifies full graph capture without implying Inductor/CUDA
    # kernel support on a Windows installation without a configured C compiler.
    compiled = torch.compile(problem.evaluate_all, backend="eager", fullgraph=True)
    actual = compiled(z)
    mapped = torch.vmap(problem.evaluate_all)(z)
    for a, b, c in zip(expected, actual, mapped, strict=True):
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(a, c)


@pytest.mark.parametrize("cls", [bilevel.TP2, bilevel.DS5] + DECEPTIVE)
def test_bilevel_follower_workflow(cls):
    problem = cls()
    xu = (problem.lb_u + problem.ub_u) / 2
    follower = problem.lower_problem(xu)
    xu.add_(1)  # Adapter must own a snapshot of the conditioning variable.
    assert not torch.equal(xu, follower.xu)
    algo = NSGA2(pop_size=12, n_objs=follower.m, lb=follower.lb, ub=follower.ub)
    workflow = UnifiedWorkflow(algo, follower)
    workflow.init_step()
    workflow.step()
    assert torch.isfinite(workflow.algorithm.fit).all()
    if problem.n_iq_l:
        assert workflow.algorithm.cv.shape == (12, problem.n_iq_l)


def test_tp_reference_fronts():
    problem = bilevel.TP1(dtype=torch.float64, ref_num=41)
    xu = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
    ps = problem.lower_ps(xu)
    torch.testing.assert_close(ps.square().sum(-1), xu.square().expand(3, 41))
    assert (ps <= 1e-15).all()
    # Each reference leader point comes from a feasible follower optimum.
    x2 = torch.linspace(-1, 0, 41, dtype=torch.float64)
    xl = torch.stack((-1 - x2, x2), -1)
    xu = xl.norm(dim=-1, keepdim=True)
    upper_f, cv = problem.evaluate_upper(xu, xl)
    torch.testing.assert_close(upper_f, problem.pf())
    assert cv.max() < 1e-14
    assert fitness(problem.evaluate_lower(xu, xl)).shape == (41, 2)
    # Negative y is allowed in TP2: its segment runs from y to zero.
    problem = bilevel.TP2(d_l=4, dtype=torch.float64, ref_num=41)
    xu = torch.tensor([[-0.8], [0.0], [1.5]], dtype=torch.float64)
    ps = problem.lower_ps(xu)
    assert ps[0, :, 0].min() == -0.8
    assert ps[0, :, 0].max() == 0
    assert (ps[..., 1:] == 0).all()
    torch.testing.assert_close(problem.lower_pf(xu), problem._objectives_lower(xu[:, None], ps))


@pytest.mark.parametrize("cls", [bilevel.DS1, bilevel.DS2, bilevel.DS3] + DECEPTIVE)
def test_ds_reference_fronts(cls):
    p = cls(dtype=torch.float64, ref_num=101)
    xu = (p.lb_u + p.ub_u) / 2
    if p.prob_id == 3:
        xu[0] = torch.round(xu[0] * 10) / 10
    ps = p.lower_ps(xu)
    assert ps.shape == (101, p.d_l)
    if p.prob_id < 3:
        assert (ps[:, 1:] == xu[1:]).all()
        front = p.lower_pf(xu)
        torch.testing.assert_close(front[:, 1], (front[:, 0].sqrt() - xu[0]).square())
    else:
        torch.testing.assert_close((ps[:, :2] - xu[:2]).square().sum(-1), torch.full((101,), p.r**2, dtype=ps.dtype))
        assert fitness(p.evaluate_lower(xu, ps)).shape == (101, 2)
    pf = p.pf()
    assert pf.ndim == 2 and pf.shape[1] == 2 and torch.isfinite(pf).all()
    if p.prob_id == 1:
        torch.testing.assert_close((pf - (1 + p.r)).square().sum(-1), torch.full((101,), (1 + p.r) ** 2, dtype=pf.dtype))
