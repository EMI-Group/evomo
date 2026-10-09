"""Deterministic regressions for previously corrected algorithm behavior."""

import importlib
import math

import pytest
import torch
from evox.core import Problem
from evox.core import compile as evox_compile

from evomo import algorithms
from evomo.algorithms import cmopso, moead_awa, prea
from evomo.workflows import UnifiedWorkflow

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class OrderedViolation(Problem):
    def evaluate(self, pop):
        value = 1 + pop.square().sum(1)
        return value[:, None].expand(-1, 2), value[:, None]


class ObjectivesOnly(Problem):
    def evaluate(self, pop):
        return torch.stack([pop.square().sum(1), (1 - pop).square().sum(1)], dim=1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("constrained", [False, True])
@pytest.mark.parametrize(
    "name", ["BiGE", "CMOPSO", "KnEA", "MaOEACSS", "CMOEA_MS", "NSGAII_SDR", "EFRRR", "TSNSGAII", "TSSparseEA", "WOF"]
)
def test_polynomial_mutation_receives_expected_mutations_per_individual(name, device, constrained, monkeypatch):
    problem = OrderedViolation() if constrained else ObjectivesOnly()
    algorithm, w = workflow(name, device, problem)
    module = importlib.import_module(getattr(algorithms, name).__module__)
    original = module.polynomial_mutation
    observed = []

    def mutation(pop, lb, ub, pro_m=1, dis_m=20):
        observed.append(pro_m)
        return original(pop, lb, ub, pro_m=pro_m, dis_m=dis_m)

    monkeypatch.setattr(module, "polynomial_mutation", mutation)
    w.step()
    assert observed and all(probability == 1 for probability in observed)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["eMOEA", "MOEADDE", "MOEAD_DCWV", "MOEADFRRMAB"])
def test_objective_only_batch_updates_remain_aligned_and_reproducible(name, device):
    states = []
    for _ in range(2):
        problem = ObjectivesOnly()
        algorithm, w = workflow(name, device, problem)
        for _ in range(12):
            w.step()
            torch.testing.assert_close(algorithm.fit, problem.evaluate(algorithm.pop))
        states.append((algorithm.pop.clone(), algorithm.fit.clone()))
    for old, new in zip(*states):
        torch.testing.assert_close(old, new, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_objective_only_moead_de_compiled_updates_remain_aligned(device):
    problem = ObjectivesOnly()
    algorithm, _ = workflow("MOEADDE", device, problem)
    algorithm.evaluate = problem.evaluate
    step = evox_compile(algorithm.step, fullgraph=True)
    for _ in range(3):
        step()
        torch.testing.assert_close(algorithm.fit, problem.evaluate(algorithm.pop))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("constrained", [False, True])
def test_epsilon_moea_pairs_population_with_archive_in_sbx_half_order(device, constrained, monkeypatch):
    from evomo.algorithms import e_moea

    problem = OrderedViolation() if constrained else ObjectivesOnly()
    algorithm, w = workflow("eMOEA", device, problem)
    algorithm.pop.fill_(0.25)
    algorithm.archive_pop.fill_(0.75)
    algorithm.archive_mask.fill_(True)
    output = problem.evaluate(algorithm.pop)
    algorithm.fit = output[0] if constrained else output
    output = problem.evaluate(algorithm.archive_pop)
    algorithm.archive_fit = output[0] if constrained else output
    if constrained:
        algorithm.cv = problem.evaluate(algorithm.pop)[1]
        algorithm.archive_cv = output[1]
    original = e_moea.simulated_binary
    seen = []

    def crossover(pop, **kwargs):
        seen.append(pop.clone())
        return original(pop, **kwargs)

    monkeypatch.setattr(e_moea, "simulated_binary", crossover)
    w.step()
    parents = seen[0]
    torch.testing.assert_close(parents[: algorithm.pop_size], torch.full_like(algorithm.pop, 0.25))
    torch.testing.assert_close(parents[algorithm.pop_size :], torch.full_like(algorithm.pop, 0.75))


def workflow(name, device, problem, size=32):
    torch.manual_seed(12)
    lb = torch.zeros(6, device=device)
    w = UnifiedWorkflow(getattr(algorithms, name)(size, 2, lb, lb + 1), problem, device=device)
    w.init_step()
    return w.algorithm, w


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("valid_count", [1, 3])
def test_smpso_leaders_never_select_padding(device, valid_count):
    algorithm, _ = workflow("SMPSO", device, OrderedViolation())
    algorithm.archive_size.fill_(valid_count)
    algorithm.archive_pop.zero_()
    algorithm.archive_pop[:valid_count] = 0.75
    algorithm.archive_fit.fill_(torch.inf)
    algorithm.archive_fit[:valid_count] = 1
    for _ in range(5):
        torch.testing.assert_close(algorithm._select_leaders(), torch.full_like(algorithm.pop, 0.75))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("constrained", [False, True])
def test_moead_awa_single_nonboundary_subproblem_keeps_index_vector(device, constrained):
    problem = OrderedViolation() if constrained else ObjectivesOnly()
    algorithm, w = workflow("MOEADAWA", device, problem)
    algorithm.W.fill_(0.5)
    boundary_count = algorithm.pop_size // 5 - 1
    algorithm.W[:boundary_count, 0] = 1
    algorithm.W[:boundary_count, 1] = 0
    w.step()
    assert len(algorithm.pop) == algorithm.pop_size


@pytest.mark.parametrize("device", DEVICES)
def test_smpso_mutation_gates_particles_without_dividing_probability_twice(device, monkeypatch):
    from evomo.algorithms import smpso

    algorithm, w = workflow("SMPSO", device, OrderedViolation(), size=128)
    observed = []

    def mutation(pop, lb, ub, pro_m=1):
        observed.append(pro_m)
        return pop

    monkeypatch.setattr(smpso, "polynomial_mutation", mutation)
    w.step()
    gate = observed[0]
    assert gate.shape == (algorithm.pop_size, 1)
    assert ((gate == 0) | (gate == 1)).all()
    assert (gate == 1).any() and (gate == 0).any()


@pytest.mark.parametrize("device", DEVICES)
def test_epsilon_moea_spreads_improving_offspring_across_incumbents(device):
    class ImprovingBatch(Problem):
        improved = False

        def evaluate(self, pop):
            value = pop.new_full((len(pop),), 0 if self.improved else 1)
            return value[:, None].expand(-1, 2), value[:, None]

    problem = ImprovingBatch()
    algorithm, w = workflow("eMOEA", device, problem)
    problem.improved = True
    w.step()
    assert (algorithm.cv == 0).sum() >= 8


@pytest.mark.parametrize("device", DEVICES)
def test_pesa2_retains_a_full_infeasible_mating_population(device):
    problem = OrderedViolation()
    algorithm, w = workflow("PESA2", device, problem)
    for _ in range(15):
        w.step()
        assert len(algorithm.pop) == algorithm.pop_size
        expected_fit, expected_cv = problem.evaluate(algorithm.pop)
        torch.testing.assert_close(algorithm.fit, expected_fit)
        torch.testing.assert_close(algorithm.cv, expected_cv)


@pytest.mark.parametrize("device", DEVICES)
def test_pesa2_grid_truncation_only_removes_cutoff_front(device):
    algorithm, _ = workflow("PESA2", device, OrderedViolation(), size=4)
    pop = torch.arange(7, device=device, dtype=torch.float32)[:, None].expand(-1, 6)
    fit = torch.ones(7, 2, device=device)
    rank = torch.tensor([0, 0, 1, 1, 1, 1, 1], device=device)
    cv = rank[:, None].float()
    selected, _ = algorithm._truncate(pop, fit, cv, rank)
    assert (selected[:, 0] == 0).any() and (selected[:, 0] == 1).any()
    assert len(selected) == 4


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scale", [1.0, 1e-15])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("name", ["MOEAD_DRA", "MOEADAWA", "MOEADDYTS", "MOEADFRRMAB"])
def test_moead_utility_rewards_constraint_improvement_despite_worse_objectives(name, device, scale, compiled):
    algorithm, _ = workflow(name, device, OrderedViolation())
    algorithm.old_obj.fill_(-1)
    algorithm.old_cv.fill_(2 * scale)
    algorithm.cv.fill_(scale)
    algorithm.pi.fill_(0.4)
    update = evox_compile(algorithm._update_utility, fullgraph=True) if compiled else algorithm._update_utility
    update()
    torch.testing.assert_close(algorithm.pi, torch.ones_like(algorithm.pi))
    torch.testing.assert_close(algorithm.old_cv, torch.full_like(algorithm.old_cv, scale))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["MOEAD_DRA", "MOEADAWA", "MOEADDYTS", "MOEADFRRMAB"])
def test_moead_utility_accepts_checkpoints_without_violation_history(name, device):
    algorithm, _ = workflow(name, device, OrderedViolation())
    state = {key: value for key, value in algorithm.state_dict().items() if key != "old_cv"}
    lb = torch.zeros(6, device=device)
    restored = getattr(algorithms, name)(32, 2, lb, lb + 1)
    restored.load_state_dict(state)
    assert restored.old_cv is None
    restored._update_utility()
    assert torch.isfinite(restored.pi).all()
    assert restored.old_cv is not None


@pytest.mark.parametrize("device", DEVICES)
def test_dmmoea_mutation_escapes_identical_real_decisions(device):
    algorithm, w = workflow("DMMOEA", device, OrderedViolation())
    algorithm.pop.fill_(0.5)
    algorithm.mask.fill_(True)
    algorithm.fit, algorithm.cv = w.problem.evaluate(algorithm.pop)
    w.step()
    assert (algorithm.pop != 0.5).any()


@pytest.mark.parametrize("device", DEVICES)
def test_dmmoea_duplicate_candidates_do_not_deadlock(device, monkeypatch):
    class Flat(Problem):
        def evaluate(self, pop):
            return pop.new_ones((len(pop), 2))

    algorithm, w = workflow("DMMOEA", device, Flat())
    algorithm.pop.fill_(0.5)
    algorithm.mask.fill_(True)
    algorithm.fit = w.problem.evaluate(algorithm.pop)
    original_rand = torch.rand

    def constant_rand(*args, **kwargs):
        return torch.full_like(original_rand(*args, **kwargs), 0.5)

    monkeypatch.setattr(torch, "rand", constant_rand)
    w.step()
    assert len(algorithm.pop) == algorithm.pop_size
    torch.testing.assert_close(algorithm.pop, torch.full_like(algorithm.pop, 0.5))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["DMMOEA", "OSP_NSDE"])
def test_odd_population_shape_and_constraint_alignment(name, device):
    problem = OrderedViolation()
    algorithm, w = workflow(name, device, problem, size=21)
    for _ in range(3):
        w.step()
        assert len(algorithm.pop) == 21
        pop = algorithm.pop * algorithm.mask if hasattr(algorithm, "mask") else algorithm.pop
        fit, cv = problem.evaluate(pop)
        torch.testing.assert_close(algorithm.fit, fit)
        torch.testing.assert_close(algorithm.cv, cv)


@pytest.mark.parametrize("device", DEVICES)
def test_osp_keeps_variation_while_all_solutions_are_infeasible(device, monkeypatch):
    from evomo.algorithms import osp_nsde

    class Infeasible(Problem):
        last = None

        def evaluate(self, pop):
            self.last = pop.clone()
            return pop.new_ones((len(pop), 2)), pop.new_ones((len(pop), 1))

    problem = Infeasible()
    algorithm, w = workflow("OSP_NSDE", device, problem)
    algorithm.t.fill_(9)
    monkeypatch.setattr(algorithm, "_arx_forecast", lambda history, p: algorithm.fit)
    monkeypatch.setattr(osp_nsde, "simulated_binary", lambda pop, **kwargs: torch.full_like(pop, 0.25))
    monkeypatch.setattr(osp_nsde, "polynomial_mutation", lambda pop, *args: pop)
    w.step()
    torch.testing.assert_close(problem.last, torch.full_like(algorithm.pop, 0.25))


class Quadratic(Problem):
    def __init__(self, constrained=False):
        super().__init__()
        self.constrained = constrained

    def evaluate(self, pop):
        value = pop.square().sum(1)
        fit = torch.stack([value, value], dim=1)
        return (fit, pop.sum(1, keepdim=True)) if self.constrained else fit


def make(name, device, constrained=False, **kwargs):
    problem = Quadratic(constrained)
    algorithm = getattr(algorithms, name)(20, 2, torch.zeros(4, device=device), torch.ones(4, device=device), **kwargs)
    workflow = UnifiedWorkflow(algorithm, problem, device=device)
    workflow.init_step()
    return workflow.algorithm, workflow


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("compiled", [False, True])
def test_prea_ratio_indicator_matches_platemo_scalar_equations(device, compiled):
    algorithm = algorithms.PREA(5, 2, torch.zeros(2, device=device), torch.ones(2, device=device))
    fit = torch.tensor([[0, 4], [1, 3], [2, 2], [1, 3], [3, 4]], device=device, dtype=torch.float32)
    algorithm.zmin = fit.amin(0, keepdim=True)
    shifted = (fit - algorithm.zmin + 1e-6).cpu()
    expected = torch.full((5, 5), torch.inf)
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            ratios = [float(shifted[j, k] / shifted[i, k] - 1) for k in range(2)]
            expected[i, j] = (
                -max(float(shifted[i, k] / shifted[j, k] - 1) for k in range(2)) if max(ratios) <= 0 else max(ratios)
            )
    function = torch.compile(algorithm._calc_indicator_matrix, fullgraph=True) if compiled else algorithm._calc_indicator_matrix
    torch.testing.assert_close(function(fit), expected.to(device))


@pytest.mark.parametrize("device", DEVICES)
def test_prea_operator_ga_half_evaluates_one_child_per_pair(device):
    algorithm, workflow = make("PREA", device)
    calls = []
    evaluate = algorithm.evaluate
    algorithm.evaluate = lambda pop: (calls.append(len(pop)), evaluate(pop))[1]
    workflow.step()
    assert calls == [algorithm.pop_size]


def scalar_prea_update(points, count):
    """Independent scalar transcription of PREA_Update's two deletion stages."""
    size, objectives = len(points), len(points[0])
    minimum = [min(point[k] for point in points) for k in range(objectives)]
    shifted = [[point[k] - minimum[k] + 1e-6 for k in range(objectives)] for point in points]
    indicator = []
    for i in range(size):
        row = []
        for j in range(size):
            value = max(shifted[j][k] / shifted[i][k] - 1 for k in range(objectives))
            if value <= 0:
                value = -max(shifted[i][k] / shifted[j][k] - 1 for k in range(objectives))
            row.append(math.inf if i == j else value)
        indicator.append(row)
    fitness = [min(row) for row in indicator]
    promising = [i for i in range(size) if fitness[i] >= 0]
    if len(promising) <= count:
        return sorted(range(size), key=lambda i: -fitness[i])[:count]
    active = promising.copy()
    while len(active) > count:
        active.remove(min(active, key=lambda i: min(indicator[i][j] for j in active)))
    maximum = [max(shifted[i][k] for i in active) for k in range(objectives)]
    active = [i for i in promising if all(shifted[i][k] <= maximum[k] for k in range(objectives))]
    normalized = [[shifted[i][k] / maximum[k] for k in range(objectives)] for i in range(size)]
    distance = [[math.inf] * size for _ in range(size)]
    for i in active:
        for j in active:
            if i == j:
                continue
            difference = [normalized[i][k] - normalized[j][k] for k in range(objectives)]
            distance[i][j] = math.sqrt(max(0, sum(v * v for v in difference) - sum(difference) ** 2 / objectives))
    while len(active) > count:
        i, j = min(((i, j) for i in active for j in active), key=lambda pair: distance[pair[0]][pair[1]])
        first = min(indicator[i][other] for other in active)
        second = min(indicator[j][other] for other in active)
        active.remove(i if first < second else j)
    return active


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("constrained", [False, True])
@pytest.mark.parametrize("objectives", [2, 3])
def test_prea_survivors_match_scalar_platemo_deletion(device, constrained, objectives, monkeypatch):
    generator = torch.Generator().manual_seed(18)
    points = torch.rand(16, objectives, generator=generator, dtype=torch.float64)
    points /= points.sum(dim=1, keepdim=True)
    points += 0.1
    expected = scalar_prea_update(points.tolist(), 8)
    points = points.to(device)
    algorithm = algorithms.PREA(8, objectives, points.new_zeros(objectives), points.new_ones(objectives))
    algorithm.pop = points[:8].clone()
    algorithm.evaluate = lambda pop: (pop, pop.new_zeros((len(pop), 1))) if constrained else pop
    algorithm.init_step()
    monkeypatch.setattr(prea, "simulated_binary", lambda *args, **kwargs: points[8:])
    monkeypatch.setattr(prea, "polynomial_mutation", lambda pop, *args: pop)
    algorithm.step()
    torch.testing.assert_close(algorithm.pop, points[expected])
    torch.testing.assert_close(algorithm.fit, points[expected])
    if constrained:
        assert torch.all(algorithm.cv == 0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("constrained", [False, True])
def test_cmopso_equal_angles_keep_first_competitor(device, constrained, monkeypatch):
    points = torch.arange(8, device=device, dtype=torch.float32).reshape(4, 2)
    algorithm = algorithms.CMOPSO(4, 2, points.new_zeros(2), points.new_full((2,), 10))
    algorithm.pop = points.clone()
    captured = []

    def evaluate(pop):
        captured.append(pop.clone())
        fit = pop.new_ones((len(pop), 2))
        return (fit, pop.new_zeros((len(pop), 1))) if constrained else fit

    algorithm.evaluate = evaluate
    algorithm.init_step()
    # Constant objectives make every competition an equal-angle tie. Front
    # crowding also ties, so leader 0 competes with leader 1 in each row.
    monkeypatch.setattr(cmopso.torch, "randint", lambda low, high, shape, **kw: torch.full(shape, low, **kw))
    monkeypatch.setattr(cmopso.torch, "rand", lambda shape, **kw: torch.full(shape, 0.5, **kw))
    monkeypatch.setattr(cmopso, "polynomial_mutation", lambda pop, *args, **kwargs: pop)
    algorithm.step()
    torch.testing.assert_close(captured[-1], points + 0.5 * (points[0] - points))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("constrained", [False, True])
def test_awa_global_mating_also_replaces_global_incumbents(device, constrained, monkeypatch):
    algorithm, workflow = make("MOEADAWA", device, constrained, nr=20, max_evaluations=100000)
    original_rand = torch.rand

    def rand(*shape, **kwargs):
        result = original_rand(*shape, **kwargs)
        return torch.full_like(result, 0.99) if result.ndim == 1 else result

    monkeypatch.setattr(moead_awa.torch, "rand", rand)
    monkeypatch.setattr(moead_awa, "simulated_binary", lambda pop: torch.zeros_like(pop))
    monkeypatch.setattr(moead_awa, "polynomial_mutation", lambda pop, lb, ub: pop)
    workflow.step()
    assert torch.all(algorithm.pop == 0)
    assert torch.all(algorithm.fit == 0)
    if constrained:
        assert torch.all(algorithm.cv == 0)


@pytest.mark.parametrize("device", DEVICES)
def test_awa_utility_and_archive_follow_evaluation_schedule(device, monkeypatch):
    algorithm, workflow = make("MOEADAWA", device, max_evaluations=2000)
    updates = []
    monkeypatch.setattr(algorithm, "_update_utility", lambda: updates.append(int(algorithm.evaluations)))
    for _ in range(12):
        workflow.step()
    assert updates == [200]
    assert int(algorithm.archive_size) == 0


@pytest.mark.parametrize("device", DEVICES)
def test_awa_preserves_weights_for_retained_subproblems(device):
    weights = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.4, 0.6], [0.7, 0.3], [0.9, 0.1]], device=device)
    fit = weights.flip(1)
    pop = torch.arange(5, device=device, dtype=weights.dtype)[:, None]
    ep_fit = torch.tensor([[0.5, 0.4]], device=device)
    ep_pop = torch.tensor([[10.0]], device=device)
    result = moead_awa._awa_logic(weights, pop, fit, ep_pop, ep_fit, 1, torch.zeros(1, 2, device=device), 2, rate=0.2)
    for retained in result[0][:-1]:
        assert (weights == retained).all(1).any()
    assert result[2][-1] == 10


@pytest.mark.parametrize("device", DEVICES)
def test_awa_does_not_insert_worse_cv_from_stale_archive(device):
    weights = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.4, 0.6], [0.7, 0.3], [0.9, 0.1]], device=device)
    pop = torch.arange(5, device=device, dtype=weights.dtype)[:, None]
    fit = weights.flip(1)
    result = moead_awa._awa_logic(
        weights,
        pop,
        fit,
        pop[:1],
        fit[:1],
        1,
        torch.zeros(1, 2, device=device),
        2,
        torch.zeros(5, 1, device=device),
        torch.ones(1, 1, device=device),
        rate=0.2,
    )
    assert torch.all(result[4] == 0)
    torch.testing.assert_close(result[0], weights)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("generation", [0, 8, 10])
def test_cmoea_ms_objective_only_follows_zero_cv_stages(device, generation):
    algorithm, _ = make("CMOEA_MS", device)
    fit = torch.tensor([[1, 4], [2, 3], [3, 2], [4, 1], [2, 2]], device=device, dtype=torch.float32)
    algorithm.iter.fill_(generation)
    torch.testing.assert_close(
        algorithm._constrained_fitness(fit, None), algorithm._constrained_fitness(fit, torch.zeros(5, 1, device=device))
    )
    if generation >= 10:
        torch.testing.assert_close(algorithm._constrained_fitness(fit, None), algorithm._cal_fitness(fit))


@pytest.mark.parametrize("device", DEVICES)
def test_cmoea_ms_invalid_constraint_scores_are_worst(device):
    algorithm, _ = make("CMOEA_MS", device)
    fit = torch.tensor([[1, 4], [2, 3], [3, 2], [4, 1]], device=device, dtype=torch.float32)
    cv = torch.tensor([[0], [0], [torch.nan], [torch.inf]], device=device)
    fitness = algorithm._constrained_fitness(fit, cv)
    assert torch.isfinite(fitness[:2]).all()
    assert torch.isposinf(fitness[2:]).all()


@pytest.mark.parametrize("device", DEVICES)
def test_knea_signed_distance_uses_distinct_maximum_extremes(device):
    algorithm, _ = make("KnEA", device)
    fit = torch.tensor([[6, 1], [1, 6], [2, 2], [4, 4], [3, 3]], device=device, dtype=torch.float32)
    knees, distance, fraction = algorithm._find_knee_points(fit, torch.tensor(0.01, device=device))
    hyperplane = torch.linalg.solve(fit[:2], torch.ones(2, 1, device=device))
    expected = -(fit @ hyperplane - 1).squeeze(1) / hyperplane.norm()
    torch.testing.assert_close(distance, expected)
    assert distance[2] > 0 and distance[3] < 0
    assert not knees[distance.argmin()]
    assert fraction == 1


@pytest.mark.parametrize("device", DEVICES)
def test_knea_first_radius_and_previous_fraction_update(device):
    algorithm, _ = make("KnEA", device)
    assert torch.all(algorithm.r == -1) and torch.all(algorithm.t == -1)
    algorithm._update_adaptive_params(0, torch.tensor(-1.0, device=device))
    assert algorithm.r[0] == 1
    algorithm._update_adaptive_params(0, torch.tensor(0.25, device=device))
    torch.testing.assert_close(algorithm.r[0], torch.exp(torch.tensor(-0.25, device=device)))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("generation,initial", [(0, False), (20, False), (20, True)])
@pytest.mark.parametrize("column_scale", [[1.0, 1.0], [1e-15, 2e-16]])
def test_cmoea_ms_platemo_stage_fitness(device, generation, initial, column_scale):
    """Independent scalar dominance oracle for both PlatEMO CMOEA-MS stages."""
    fit = torch.tensor([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0], [0.0, 0.0]], device=device)
    cv = torch.tensor([[0.0, -2.0], [0.0, 0.0], [0.0, 0.0], [1.0, 3.0], [2.0, 1.0]], device=device)
    cv = cv * torch.tensor(column_scale, device=device)
    bounds = torch.zeros(2, device=device)
    algorithm = algorithms.CMOEA_MS(5, 2, bounds, bounds + 1, max_gen=100)
    algorithm.iter.fill_(generation)
    normalized = (fit - fit.amin(0)) / (fit.amax(0) - fit.amin(0)).clamp_min(1e-12)
    maximum = cv.clamp_min(0).amax(0)
    totals = (cv.clamp_min(0) / torch.where(maximum > 0, maximum, torch.ones_like(maximum))).mean(1)
    k = int(len(fit) ** 0.5)
    sde = []
    for i in range(len(fit)):
        distances = [
            torch.linalg.vector_norm(normalized[i] - torch.maximum(normalized[i], normalized[j]))
            for j in range(len(fit))
            if j != i
        ]
        sde.append(1 / (torch.stack(distances).sort().values[k - 1] + 2))
    late = generation + 1 >= 10 and not initial
    objectives = fit if late else torch.stack([torch.stack(sde), totals], dim=1)
    constraints = totals if late else torch.zeros_like(totals)
    dom = torch.zeros((len(fit), len(fit)), dtype=torch.bool, device=device)
    for i in range(len(fit)):
        for j in range(len(fit)):
            dom[i, j] = (constraints[i] < constraints[j]) | (
                (constraints[i] == constraints[j])
                & (objectives[i] <= objectives[j]).all()
                & (objectives[i] < objectives[j]).any()
            )
    raw = dom.T.to(fit.dtype) @ dom.sum(1).to(fit.dtype)
    density = []
    for i in range(len(fit)):
        distances = [
            1 - torch.nn.functional.cosine_similarity(objectives[i], objectives[j], dim=0) for j in range(len(fit)) if j != i
        ]
        density.append(1 / (torch.stack(distances).sort().values[k - 1] + 2))
    expected = raw + torch.stack(density)
    torch.testing.assert_close(algorithm._constrained_fitness(fit, cv, initial=initial), expected)


@pytest.mark.parametrize("device", DEVICES)
def test_commea_keeps_full_second_population_under_tight_epsilon_filter(device):
    class OrderedProblem(Problem):
        def evaluate(self, pop):
            value = 1 + pop.square().sum(1)
            return value[:, None].expand(-1, 2), value[:, None]

    problem = OrderedProblem()
    algorithm, w = workflow("CoMMEA", device, problem)
    for _ in range(25):
        w.step()
        assert len(algorithm.pop2) == algorithm.pop_size
        assert_population_state(problem, algorithm.pop2, algorithm.fit2, algorithm.cv2)


@pytest.mark.parametrize("device", DEVICES)
def test_bce_moead_skips_empty_exploration_evaluation(device, monkeypatch):
    class NonemptyProblem(OrderedViolation):
        def evaluate(self, pop):
            assert pop.shape[0] > 0
            return super().evaluate(pop)

    problem = NonemptyProblem()
    algorithm, w = workflow("BCEMOEAD", device, problem)
    monkeypatch.setattr(algorithm, "_exploration", lambda: algorithm.pop[:0])
    w.step()
    assert_population_state(problem, algorithm.pop, algorithm.fit, algorithm.cv)
    assert_population_state(problem, algorithm.npc_pop, algorithm.npc_fit, algorithm.npc_cv)


def assert_population_state(problem, pop, fit, cv, mask=None):
    if mask is not None:
        pop = pop * mask
    expected_fit, expected_cv = problem.evaluate(pop)
    valid = torch.isfinite(pop).all(1)
    assert valid.any()
    torch.testing.assert_close(expected_fit[valid], fit[valid])
    torch.testing.assert_close(expected_cv[valid], cv[valid])
    assert cv.device == pop.device
    assert cv.dtype == expected_cv.dtype
