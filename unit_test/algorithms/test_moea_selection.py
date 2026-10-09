"""Independent selection oracles and algorithm-specific constraint invariants."""

import pytest
import torch
from evox.core import Problem

from evomo.algorithms import IBEA, NSGA3, HypE, RVEAa, TensorMOEAD
from evomo.algorithms.hype import cal_hv
from evomo.algorithms.nsga3 import _constrained_rank, _normalize
from evomo.operators.selection.constraint_handling import total_violation
from evomo.problems.numerical import DTLZ2
from evomo.workflows import UnifiedWorkflow

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
AGGREGATIONS = [
    ("pbi", "pbi"),
    ("tchebycheff", "tchebycheff"),
    ("tchebycheff_norm", "tchebycheff_norm"),
    ("modified_tchebycheff", "modified_tchebycheff"),
    ("weighted_sum", "weighted_sum"),
    ("tchebycheff", "weighted_sum"),
]


def reference_ranks(fit, cv):
    """Independent small-loop constrained-dominance oracle, zero-based."""
    values = fit.cpu().tolist()
    positive = cv.clamp_min(0)
    total = (positive.sum(1) if cv.ndim == 2 else positive).cpu().tolist()
    n = len(values)
    domination = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if total[i] == 0 and total[j] == 0:
                domination[i][j] = all(a <= b for a, b in zip(values[i], values[j])) and any(
                    a < b for a, b in zip(values[i], values[j])
                )
            else:
                domination[i][j] = total[i] < total[j]
    rank = [-1] * n
    level = 0
    while -1 in rank:
        front = [j for j in range(n) if rank[j] == -1 and not any(rank[i] == -1 and domination[i][j] for i in range(n))]
        assert front
        for j in front:
            rank[j] = level
        level += 1
    return torch.tensor(rank, device=fit.device, dtype=torch.int32)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", ["mixed", "feasible", "infeasible", "scalar", "empty", "constant"])
def test_rank_matches_platemo_rules(device, case):
    fit = torch.tensor([[4.0, 4], [0, 0], [1, 1], [2, 2], [3, 3], [1, 1]], device=device)
    cv = torch.tensor([[0.0, -1], [1, 0], [1, 0], [2, 0], [0.2, -3], [0, 0]], device=device)
    if case == "feasible":
        cv.zero_()
    elif case == "infeasible":
        cv[:, 0] += 1
    elif case == "scalar":
        cv = cv[:, 0]
    elif case == "empty":
        cv = cv[:, :0]
    elif case == "constant":
        fit.fill_(1)
    expected = reference_ranks(fit, cv)
    torch.testing.assert_close(_constrained_rank(fit, cv), expected)
    partial = _constrained_rank(fit, cv, 3)
    cutoff = expected.kthvalue(3).values
    torch.testing.assert_close(partial[expected <= cutoff], expected[expected <= cutoff])
    assert (partial[expected > cutoff] == len(fit)).all()


@pytest.mark.parametrize("device", DEVICES)
def test_niching_keeps_complete_fronts_and_equal_cv_ties(device):
    algo = NSGA3(4, 2, torch.zeros(2, device=device), torch.ones(2, device=device), device=device)
    fit = torch.tensor([[20.0, 20], [10, 10], [0, 0], [1, 1], [2, 2], [3, 3]], device=device)
    cv = torch.tensor([[0.0], [0.0], [1], [1], [1], [2]], device=device)
    algo.z_min = fit.new_tensor([10, 10])
    selected = algo._survive(fit, cv)
    assert len(selected.unique()) == 4
    assert 0 in selected and 1 in selected
    assert 5 not in selected
    assert ((selected >= 2) & (selected <= 4)).sum() == 2
    expected = reference_ranks(fit, cv)
    torch.testing.assert_close(algo.rank, expected[selected])
    # Objective dominance must not split the three equal-CV infeasible points.
    assert expected[2] == expected[3] == expected[4]


class ControlledProblem(Problem):
    def __init__(self):
        super().__init__()
        self.cv_value = 1.0
        self.fit_value = 2.0

    def evaluate(self, x):
        return x.new_full((len(x), 2), self.fit_value), x.new_full((len(x), 1), self.cv_value)


@pytest.mark.parametrize("device", DEVICES)
def test_feasible_ideal_history_and_mating_key(device):
    recorded = []

    def selection(count, keys):
        recorded.append(keys[0].clone())
        return torch.arange(count, device=device)

    problem = ControlledProblem()
    algo = NSGA3(6, 2, torch.zeros(3, device=device), torch.ones(3, device=device), selection_op=selection, device=device)
    workflow = UnifiedWorkflow(algo, problem, device=device)
    workflow.init_step()
    state = workflow.algorithm
    assert torch.isinf(state.z_min).all()
    workflow.step()
    torch.testing.assert_close(recorded[-1], state.fit.new_ones(6))
    assert torch.isinf(state.z_min).all()
    problem.cv_value, problem.fit_value = 0.0, 5.0
    workflow.step()
    torch.testing.assert_close(state.z_min, state.fit.new_full((2,), 5))
    # Better objectives from infeasible offspring cannot move the feasible ideal.
    problem.cv_value, problem.fit_value = 1.0, -100.0
    workflow.step()
    torch.testing.assert_close(recorded[-1], state.fit.new_zeros(6))
    torch.testing.assert_close(state.z_min, state.fit.new_full((2,), 5))
    assert torch.isfinite(state.fit).all()
    restored = NSGA3(6, 2, algo.lb, algo.ub, device=device)
    restored.load_state_dict(state.state_dict())
    torch.testing.assert_close(restored.cv, state.cv)
    torch.testing.assert_close(restored.z_min, state.z_min)


@pytest.mark.parametrize("device", DEVICES)
def test_normalization_external_ideal_and_degeneracy(device):
    fit = torch.ones(8, 3, device=device, dtype=torch.float64)
    ideal = fit.new_zeros(3)
    normalized = _normalize(fit, ideal=ideal)
    assert torch.isfinite(normalized).all()
    torch.testing.assert_close(normalized, fit)
    torch.testing.assert_close(_normalize(fit), torch.zeros_like(fit))


@pytest.mark.parametrize("device", DEVICES)
def test_unconstrained_fullgraph(device):
    torch.manual_seed(8)
    workflow = UnifiedWorkflow(
        NSGA3(12, 3, torch.zeros(5, device=device), torch.ones(5, device=device), device=device),
        DTLZ2(d=5, m=3),
        device=device,
    )
    workflow.init_step()
    step = torch.compile(workflow.step, fullgraph=True)
    for _ in range(3):
        step()
    assert workflow.algorithm.cv is None and workflow.algorithm.z_min is None
    assert torch.isfinite(workflow.algorithm.fit).all()


class TensorRowsProblem(Problem):
    def __init__(self, cv_kind="multi"):
        super().__init__()
        self.cv_kind = cv_kind

    def evaluate(self, x):
        if self.cv_kind == "scalar":
            cv = x[:, 2]
        elif self.cv_kind == "empty":
            cv = x[:, 2:2]
        else:
            cv = x[:, 2:]
        return x[:, :2], cv


def tensor_reference_update(algorithm, offspring, problem):
    """Independent loop oracle preserving the original two-stage candidate order."""
    pop, fit, cv = algorithm.pop, algorithm.fit, algorithm.cv
    off_fit, off_cv = problem.evaluate(offspring)
    old_total, new_total = total_violation(cv), total_violation(off_cv)
    z = torch.minimum(algorithm.z, off_fit.amin(0))
    z_max = torch.maximum(fit.amax(0), off_fit.amax(0))
    selected_pop, selected_fit, selected_cv = [], [], []
    for target in range(len(pop)):
        candidates = []
        for source in range(len(offspring)):
            g_old = algorithm.aggregate_func1(fit[target : target + 1], algorithm.w[target : target + 1], z, z_max)[0]
            g_new = algorithm.aggregate_func1(off_fit[source : source + 1], algorithm.w[target : target + 1], z, z_max)[0]
            eligible = target in algorithm.neighbors[source] and (
                new_total[source] < old_total[target] or (new_total[source] == old_total[target] and g_new < g_old)
            )
            if eligible:
                candidates.append((offspring[source], off_fit[source], off_cv[source], new_total[source]))
            else:
                candidates.append((pop[target], fit[target], cv[target], old_total[target]))
        scores = algorithm.aggregate_func2(torch.stack([c[1] for c in candidates]), algorithm.w[target : target + 1], z, z_max)
        best = min(range(len(candidates)), key=lambda i: (float(candidates[i][3]), float(scores[i]), i))
        x, f, c, _ = candidates[best]
        selected_pop.append(x)
        selected_fit.append(f)
        selected_cv.append(c)
    return torch.stack(selected_pop), torch.stack(selected_fit), torch.stack(selected_cv), z, z_max


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("aggregation", AGGREGATIONS)
@pytest.mark.parametrize("case", ["mixed", "feasible", "infeasible", "equal", "constant", "scalar", "empty"])
def test_tensor_moead_two_stage_update_matches_reference(device, aggregation, case):
    torch.manual_seed(12)
    lb, ub = torch.full((4,), -1.0, device=device), torch.full((4,), 2.0, device=device)
    pop, children = torch.rand(12, 4, device=device), torch.rand(12, 4, device=device)
    pop[:3] = pop[0].clone()
    if case == "feasible":
        pop[:, 2:], children[:, 2:] = -0.5, -0.2
    elif case == "infeasible":
        pop[:, 2:] += 0.1
        children[:, 2:] += 0.1
    elif case == "equal":
        pop[:, 2:], children[:, 2:] = 0.5, 0.5
    elif case == "constant":
        pop[:, :2], children[:, :2] = 0.5, 0.5
    else:
        pop[:, 2:] -= 0.5
        children[:, 2:] -= 0.5
    problem = TensorRowsProblem(case)
    algo = TensorMOEAD(
        12,
        2,
        lb,
        ub,
        aggregate_op=aggregation,
        crossover_op=lambda _: children,
        mutation_op=lambda x, lb, ub: x,
        device=device,
    )
    algo.pop = pop
    workflow = UnifiedWorkflow(algo, problem, device=device)
    workflow.init_step()
    state = workflow.algorithm
    expected = tensor_reference_update(state, children, problem)
    workflow.step()
    for actual, want in zip((state.pop, state.fit, state.cv, state.z, state.z_max), expected):
        torch.testing.assert_close(actual, want)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("compiled", [False, True])
def test_tensor_moead_feasible_child_wins_despite_worse_objective(device, compiled):
    # The original stage-two argmin would retain the infeasible incumbent here.
    pop = torch.tensor([0.0, 0.0, 0.5, -0.8], device=device).expand(12, -1).clone()
    children = torch.tensor([1.0, 1.0, 0.0, -0.8], device=device).expand(12, -1).clone()
    algo = TensorMOEAD(
        12,
        2,
        pop.new_full((4,), -1),
        pop.new_full((4,), 2),
        aggregate_op=("weighted_sum", "weighted_sum"),
        crossover_op=lambda _: children,
        mutation_op=lambda x, lb, ub: x,
        device=device,
    )
    algo.pop = pop
    workflow = UnifiedWorkflow(algo, TensorRowsProblem(), device=device)
    workflow.init_step()
    step = torch.compile(workflow.step, fullgraph=True) if compiled else workflow.step
    step()
    torch.testing.assert_close(workflow.algorithm.pop, children)
    assert (total_violation(workflow.algorithm.cv) == 0).all()


class IndicatorRowsProblem(Problem):
    def __init__(self, kind="multi"):
        super().__init__()
        self.kind = kind

    def evaluate(self, x):
        cv = x[:, 3] if self.kind == "scalar" else x[:, 3:3] if self.kind == "empty" else x[:, 3:]
        return x[:, :2], cv


def make_indicator_algorithm(cls, lb, ub, size=12, n_objs=2, **kwargs):
    if cls is HypE:
        kwargs["n_sample"] = 64
    return cls(n_objs=n_objs, pop_size=size, lb=lb, ub=ub, device=lb.device, **kwargs)


def ibea_reference(fit, cv, count, kappa=0.05):
    """Small-loop deletion oracle with a separate active-index list."""
    span = fit.amax(0) - fit.amin(0)
    norm = (fit - fit.amin(0)) / torch.where(span > 0, span, 1)
    indicator = (norm[:, None, :] - norm[None, :, :]).amax(2)
    scale = indicator.abs().amax(0)
    scale = torch.where(scale > 0, scale, 1)
    scores = -torch.exp(-indicator / scale[None, :] / kappa).sum(0) + 1
    totals = total_violation(cv).cpu().tolist()
    active = list(range(len(fit)))
    while len(active) > count:
        worst = min(active, key=lambda i: (-totals[i], float(scores[i]), i))
        scores += torch.exp(-indicator[worst] / scale[worst] / kappa)
        active.remove(worst)
    return torch.tensor(active, device=fit.device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["mixed", "feasible", "infeasible", "equal", "constant", "scalar", "empty"])
def test_ibea_matches_independent_deletion(device, kind):
    torch.manual_seed(17)
    rows = torch.rand(24, 5, device=device, dtype=torch.float64)
    rows[:, 2] = torch.arange(24, device=device) / 24
    rows[:, 3:] -= 0.5
    rows[:2, :2] = rows[0, :2].clone()
    if kind == "feasible":
        rows[:, 3:] = -0.5
    elif kind == "infeasible":
        rows[:, 3:] = rows[:, 3:].abs() + 0.1
    elif kind == "equal":
        rows[:, 3:] = 0.5
    elif kind == "constant":
        rows[:, :2] = 1
    lb, ub = rows.new_full((5,), -1), rows.new_full((5,), 2)
    algo = make_indicator_algorithm(IBEA, lb, ub, crossover_op=lambda _: rows[12:], mutation_op=lambda x, lb, ub: x)
    algo.pop = rows[:12].clone()
    workflow = UnifiedWorkflow(algo, IndicatorRowsProblem(kind), device=device)
    workflow.init_step()
    fit, cv = workflow.problem.evaluate(rows)
    expected = ibea_reference(fit, cv, 12)
    workflow.step()
    torch.testing.assert_close(workflow.algorithm.pop, rows[expected])
    torch.testing.assert_close(workflow.algorithm.cv, cv[expected])
    assert workflow.algorithm.fit.dtype == torch.float64


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("compiled", [False, True])
def test_masked_hv_matches_compact_front(device, compiled):
    fit = torch.tensor([[0.1, 0.8], [0.9, 0.9], [0.5, 0.5], [0.8, 0.1], [0.0, 0.0]], device=device)
    valid = torch.tensor([True, False, True, True, False], device=device)
    ref = fit.new_tensor([1.2, 1.2])
    # Fixed random seed checks masking removes both dominance and sampling-bound effects.
    fn = torch.compile(cal_hv, fullgraph=True) if compiled else cal_hv
    torch.manual_seed(23)
    masked = fn(fit, ref, 1, 128, valid)
    torch.manual_seed(23)
    compact = fn(fit[valid], ref, 1, 128)
    torch.testing.assert_close(masked[valid], compact)
    assert (masked[~valid] == 0).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cls", [IBEA, HypE])
@pytest.mark.parametrize("compiled", [False, True])
def test_feasible_survival_and_mating_priority(device, cls, compiled):
    # Feasible offspring have worse objectives than every infeasible incumbent.
    pop = torch.tensor([0.1, 0.1, 0.0, 0.5, -0.8], device=device).expand(12, -1).clone()
    children = torch.tensor([0.9, 0.9, 1.0, 0.0, -0.8], device=device).expand(12, -1).clone()
    algo = make_indicator_algorithm(
        cls, pop.new_full((5,), -1), pop.new_full((5,), 2), crossover_op=lambda _: children, mutation_op=lambda x, lb, ub: x
    )
    algo.pop = pop
    workflow = UnifiedWorkflow(algo, IndicatorRowsProblem(), device=device)
    workflow.init_step()
    step = torch.compile(workflow.step, fullgraph=True) if compiled else workflow.step
    step()
    torch.testing.assert_close(workflow.algorithm.pop, children)
    assert (total_violation(workflow.algorithm.cv) == 0).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["scalar", "empty", "constant", "equal"])
def test_hype_optional_shapes_and_degeneracy(device, kind):
    torch.manual_seed(21)
    rows = torch.rand(24, 5, device=device)
    rows[:, 2] = torch.arange(24, device=device) / 24
    if kind == "constant":
        rows[:, :2] = 1
    if kind == "equal":
        rows[:, 3:] = 0.5
    algo = make_indicator_algorithm(
        HypE, rows.new_zeros(5), rows.new_full((5,), 2), crossover_op=lambda _: rows[12:], mutation_op=lambda x, lb, ub: x
    )
    algo.pop = rows[:12].clone()
    workflow = UnifiedWorkflow(algo, IndicatorRowsProblem(kind), device=device)
    workflow.init_step()
    workflow.step()
    state = workflow.algorithm
    fit, cv = workflow.problem.evaluate(state.pop)
    torch.testing.assert_close(state.fit, fit)
    torch.testing.assert_close(state.cv, cv)
    assert torch.isfinite(state.fit).all()
    assert (total_violation(cv) <= total_violation(workflow.problem.evaluate(rows)[1]).kthvalue(12).values).all()


@pytest.mark.parametrize("device", DEVICES)
def test_final_truncation_prefers_lower_violation_and_preserves_slots(device):
    algo = RVEAa(12, 2, torch.zeros(3, device=device), torch.ones(3, device=device), device=device)
    pop = torch.rand(24, 3, device=device)
    fit = torch.rand(24, 2, device=device)
    cv = torch.arange(24, device=device, dtype=fit.dtype)[:, None].flip(0)
    pop[0], fit[0], cv[0] = torch.nan, torch.nan, torch.nan
    new_pop, new_fit, new_cv = algo._constrained_batch_truncation(pop, fit, cv)
    kept = torch.isfinite(new_fit).all(1)
    assert kept.sum() == 12
    assert not kept[:12].any() and kept[12:].all()
    torch.testing.assert_close(new_pop[kept], pop[kept])
    torch.testing.assert_close(new_cv[kept], cv[kept])
    assert torch.isnan(new_cv[~kept]).all()


class FeasibleWorseObjectives(Problem):
    def evaluate(self, x):
        cv = (x[:, :1] < 0.5).to(x.dtype)
        return torch.where(cv.bool(), 0.1, 0.9).expand(-1, 2), cv


@pytest.mark.parametrize("device", DEVICES)
def test_first_front_does_not_drop_objectively_dominated_feasible_points(device):
    algo = RVEAa(
        12,
        2,
        torch.zeros(3, device=device),
        torch.ones(3, device=device),
        device=device,
        mutation_op=lambda x, lb, ub: x,
        crossover_op=lambda x: x,
    )
    algo.pop[:6, 0] = 0.1
    algo.pop[6:, 0] = 0.9
    workflow = UnifiedWorkflow(algo, FeasibleWorseObjectives(), device=device)
    workflow.init_step()
    workflow.step()
    valid = torch.isfinite(workflow.algorithm.fit).all(1)
    assert valid.any()
    # Feasibility is local to a vector partition; other partitions may retain
    # infeasible points, but the objective dominator cannot erase all feasible ones.
    assert (total_violation(workflow.algorithm.cv[valid]) == 0).any()


class InfeasibleTradeoffs(Problem):
    def evaluate(self, x):
        return torch.stack((x[:, 0], 1 - x[:, 0]), 1), 0.1 + x[:, :1]


@pytest.mark.parametrize("device", DEVICES)
def test_infeasible_first_front_keeps_objective_violation_tradeoffs(device):
    algo = RVEAa(
        12,
        2,
        torch.zeros(3, device=device),
        torch.ones(3, device=device),
        device=device,
        crossover_op=lambda x: x,
        mutation_op=lambda x, lb, ub: x,
    )
    algo.pop[:, 0] = torch.linspace(0.1, 0.9, 12, device=device)
    workflow = UnifiedWorkflow(algo, InfeasibleTradeoffs(), device=device)
    workflow.init_step()
    workflow.step()
    valid = torch.isfinite(workflow.algorithm.fit).all(1)
    assert valid.sum() > 1
    assert workflow.algorithm.pop[valid, 0].unique().numel() > 1
