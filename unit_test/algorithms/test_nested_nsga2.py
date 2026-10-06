"""NestedNSGA2 end-to-end search and compiled execution."""

import copy

import pytest
import torch

from evomo.algorithms import NestedNSGA2
from evomo.operators.selection import non_dominate as nd_ops
from evomo.problems.bilevel import DS3D, TP1, TP2

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))]


def repair_ds3(xu):
    return torch.cat((torch.round(10 * xu[..., :1]) / 10, xu[..., 1:]), -1)


def assert_state(a):
    p = a.problem
    actual = p.evaluate_all(a.pop)
    torch.testing.assert_close(a.fit, actual.upper_fitness)
    torch.testing.assert_close(a.lower_fit, actual.lower_fitness)
    torch.testing.assert_close(a.cv[:, 1:], torch.cat((actual.upper_cv, actual.lower_cv), -1))
    assert a.pop.dtype == p.dtype and a.pop.device == p.device
    assert ((a.pop >= p.lb) & (a.pop <= p.ub)).all()
    assert torch.isfinite(a.fit).all() and torch.isfinite(a.lower_fit).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["TP1", "TP2", "DS3D"])
def test_nested_nsga2(device, dtype, kind):
    torch.manual_seed(12)
    p = {"TP1": TP1, "TP2": lambda **kw: TP2(d_l=2, **kw), "DS3D": lambda **kw: DS3D(k=4, **kw)}[kind](
        device=device, dtype=dtype
    )
    a = NestedNSGA2(p, pop_size=8, lower_pop_size=8, lower_generations=2, upper_repair=repair_ds3 if kind == "DS3D" else None)
    with pytest.raises(RuntimeError, match="init_step"):
        a.step()
    a.init_step()
    with pytest.raises(RuntimeError, match="already initialized"):
        a.init_step()
    for _ in range(2):
        a.step()
        assert_state(a)
    assert a.generation == 2
    assert a.upper_evaluations == 3 * 8 * 8
    assert a.lower_evaluations == 3 * 8 * 8 * 3
    if kind == "DS3D":
        torch.testing.assert_close(a.pop[:, 0] * 10, torch.round(a.pop[:, 0] * 10))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("constrained", [False, True])
def test_nested_nsga2_compiled(device, constrained, monkeypatch):
    # Isolate graph capture from the ranking loop's separate Inductor compiler.
    # CPU capture should not require a platform-specific C++ toolchain.
    if device == "cpu":
        monkeypatch.setattr(
            nd_ops,
            "_partial_rank_compile",
            torch.compile(nd_ops._partial_rank_compile._torchdynamo_orig_callable, backend="eager", fullgraph=True),
        )
    p = TP1(device=device, dtype=torch.float64) if constrained else TP2(d_l=2, device=device, dtype=torch.float64)
    a = NestedNSGA2(p, pop_size=4, lower_pop_size=4, lower_generations=1)
    a.init_step()
    b = copy.deepcopy(a)
    torch.manual_seed(24)
    a.step()
    torch.manual_seed(24)
    torch.compile(b.step, backend="eager", fullgraph=True)()
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key])
    assert_state(b)
