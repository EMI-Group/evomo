"""Core constraint state, feasibility and compiled execution checks."""

import sys

import pytest
import torch
from evox.core import Problem, use_state, vmap
from evox.core import compile as evox_compile

from evomo import algorithms
from evomo.operators.selection.constraint_handling import total_violation
from evomo.workflows import UnifiedWorkflow

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
SPECIALIZED = ("NSGA3", "RVEA", "RVEAa", "IBEA", "HypE", "TensorMOEAD")
ALGORITHMS = SPECIALIZED + (
    "AGEMOEA",
    "BCE_IBEA",
    "BCEMOEAD",
    "BiGE",
    "CLIA",
    "CMOEA_MS",
    "CMOPSO",
    "CoMMEA",
    "DMMOEA",
    "EFRRR",
    "eMOEA",
    "GDE3",
    "GrEA",
    "GWASFGA",
    "KnEA",
    "LMOCSO",
    "LSMOF",
    "MaOEACSS",
    "MOEAD",
    "MOEADAWA",
    "MOEAD_DCWV",
    "MOEADDE",
    "MOEAD_DRA",
    "MOEADDU",
    "MOEADDYTS",
    "MOEADFRRMAB",
    "MOEAD_PaS",
    "MOEAURAW",
    "NSBiDiCo",
    "NSGAII_SDR",
    "OSP_NSDE",
    "PESA2",
    "PICEAg",
    "PREA",
    "SIBEA",
    "SMPSO",
    "SNSGA2",
    "SparseEA",
    "SparseEA2",
    "SPEAR",
    "SSCEA",
    "tDEA_CPBI",
    "TELSO",
    "ThetaDEA",
    "TSNSGAII",
    "TSSparseEA",
    "Two_Arch2",
    "VaEA",
    "WASFGA",
    "WOF",
)


class ConstrainedToy(Problem):
    def __init__(self, mode="signed", constant=False):
        super().__init__()
        self.mode = mode
        self.constant = constant

    def evaluate(self, pop):
        fit = torch.stack([pop.square().sum(1), (pop - 1).square().sum(1)], dim=1)
        if self.constant:
            fit = torch.ones_like(fit)
        if self.mode == "signed":
            cv = torch.stack([0.45 - pop[:, 0], -torch.ones_like(pop[:, 0])], dim=1)
        elif self.mode == "infeasible":
            cv = pop[:, 0] + 1
        elif self.mode == "tied":
            cv = torch.ones_like(pop[:, :1])
        elif self.mode == "positive":
            cv = pop[:, :1] + 1
        elif self.mode == "worse":
            fit = torch.zeros_like(fit)
            cv = torch.full_like(pop[:, :1], 3)
        elif self.mode == "empty":
            cv = pop.new_empty((pop.shape[0], 0))
        else:
            cv = torch.zeros_like(pop[:, :1])
        return fit, cv


def make_workflow(name, device, problem):
    torch.manual_seed(42)
    bounds = torch.zeros(6, device=device)
    kwargs = {"device": torch.device(device)} if name in {"MOEAD", "LMOCSO", *SPECIALIZED} else {}
    if name == "HypE":
        kwargs["n_sample"] = 64
    if name in {"RVEA", "RVEAa"}:
        kwargs.update(fr=0.5, max_gen=3)
    algorithm = getattr(algorithms, name)(pop_size=20, n_objs=2, lb=bounds, ub=bounds + 1, **kwargs)
    workflow = UnifiedWorkflow(algorithm, problem, device=torch.device(device))
    workflow.init_step()
    return algorithm, workflow


def assert_population_state(problem, pop, fit, cv, mask=None):
    if mask is not None:
        pop = pop * mask
    expected_fit, expected_cv = problem.evaluate(pop)
    valid = torch.isfinite(pop).all(1) & torch.isfinite(fit).all(1)
    assert valid.any()
    torch.testing.assert_close(expected_fit[valid], fit[valid])
    torch.testing.assert_close(expected_cv[valid], cv[valid])
    assert cv.device == pop.device
    assert cv.dtype == expected_cv.dtype


@pytest.mark.parametrize("name", ALGORITHMS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["signed", "infeasible", "empty"])
def test_constraint_state_alignment(name, device, mode):
    problem = ConstrainedToy(mode)
    algorithm, workflow = make_workflow(name, device, problem)
    for _ in range(3):
        workflow.step()
    assert_population_state(problem, algorithm.pop, algorithm.fit, algorithm.cv, getattr(algorithm, "mask", None))
    if name in {"RVEA", "RVEAa"}:
        invalid = ~torch.isfinite(algorithm.fit).all(1)
        assert torch.isnan(algorithm.cv[invalid]).all()
    for pop_name, fit_name, cv_name in (
        ("npc_pop", "npc_fit", "npc_cv"),
        ("pop2", "fit2", "cv2"),
        ("pbest_pop", "pbest_fit", "pbest_cv"),
        ("archive_pop", "archive_fit", "archive_cv"),
        ("archive", "archive_fit", "archive_cv"),
    ):
        if hasattr(algorithm, cv_name) and isinstance(getattr(algorithm, pop_name, None), torch.Tensor):
            pop, fit, cv = (getattr(algorithm, key) for key in (pop_name, fit_name, cv_name))
            if cv is None or not pop.shape[0]:
                continue
            if hasattr(algorithm, "archive_mask") and cv_name == "archive_cv":
                valid = algorithm.archive_mask
            elif hasattr(algorithm, "archive_size") and cv_name == "archive_cv":
                valid = torch.arange(pop.shape[0], device=pop.device) < algorithm.archive_size
            else:
                valid = torch.isfinite(fit).all(1)
            if valid.any():
                assert_population_state(problem, pop[valid], fit[valid], cv[valid])


@pytest.mark.parametrize("name", ALGORITHMS)
@pytest.mark.parametrize("device", DEVICES)
def test_constant_objectives(name, device):
    problem = ConstrainedToy("tied", constant=True)
    algorithm, workflow = make_workflow(name, device, problem)
    for _ in range(3):
        workflow.step()
    assert_population_state(problem, algorithm.pop, algorithm.fit, algorithm.cv, getattr(algorithm, "mask", None))
    if name in {"RVEA", "RVEAa"}:
        assert torch.isfinite(algorithm.reference_vector).all()
        assert (algorithm.reference_vector.norm(dim=1) > 0).all()


@pytest.mark.parametrize("name", [n for n in ALGORITHMS if n not in {"SMPSO", "LMOCSO", "TELSO", "CMOEA_MS", "RVEA", "RVEAa"}])
@pytest.mark.parametrize("device", DEVICES)
def test_infeasible_offspring_cannot_displace_feasible_incumbents(name, device):
    problem = ConstrainedToy("zero")
    algorithm, workflow = make_workflow(name, device, problem)
    problem.mode = "tied"
    workflow.step()
    valid = torch.isfinite(algorithm.fit).all(1)
    assert (total_violation(algorithm.cv)[valid] == 0).all()


@pytest.mark.parametrize("name", [n for n in ALGORITHMS if n not in {"SMPSO", "LMOCSO", "TELSO", "CMOEA_MS", "RVEA", "RVEAa"}])
@pytest.mark.parametrize("device", DEVICES)
def test_lower_violation_beats_dominating_infeasible_offspring(name, device):
    problem = ConstrainedToy("positive")
    algorithm, workflow = make_workflow(name, device, problem)
    problem.mode = "worse"
    workflow.step()
    valid = torch.isfinite(algorithm.fit).all(1)
    assert (total_violation(algorithm.cv)[valid] < 3).all()


def test_public_algorithm_constraint_coverage():
    handled_before = {"NSGA2"}
    assert set(ALGORITHMS) | handled_before == set(algorithms.__all__)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["EFRRR", "LMOCSO", "MOEADDE", "MOEADDU", *SPECIALIZED])
def test_compiled_constraint_state(name, device, request):
    if name == "EFRRR" and device == "cpu" and sys.platform == "win32" and torch.__version__.startswith("2.11."):
        request.node.add_marker(
            pytest.mark.xfail(
                reason="Windows CPU Inductor returns int64 fitness; reproduced in the original unconstrained EFRRR too",
                strict=True,
            )
        )
    torch.compiler.reset()
    problem = ConstrainedToy()
    algorithm, workflow = make_workflow(name, device, problem)
    algorithm.evaluate = problem.evaluate
    step = evox_compile(algorithm.step, fullgraph=True)
    for _ in range(3):
        step()
    assert_population_state(problem, algorithm.pop, algorithm.fit, algorithm.cv)
    torch.compiler.reset()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", SPECIALIZED)
def test_constrained_state_roundtrip_and_compiled_vmap(name, device):
    problem = ConstrainedToy()
    algorithm, workflow = make_workflow(name, device, problem)
    workflow.step()  # RVEAa establishes its fixed 2*N slot layout here.
    restored, _ = make_workflow(name, device, problem)
    restored.pop = torch.empty_like(algorithm.pop)
    restored.fit = torch.empty_like(algorithm.fit)
    restored.cv = torch.empty_like(algorithm.cv)
    restored.load_state_dict(algorithm.state_dict())
    torch.testing.assert_close(restored.cv, algorithm.cv, equal_nan=True)
    params, buffers = torch.func.stack_module_state([workflow, workflow])
    step = torch.compile(vmap(use_state(workflow.step), randomness="different"), fullgraph=True)
    state = params | buffers
    for _ in range(2):
        state = step(state)
    for batch in range(2):
        assert_population_state(
            problem, state["algorithm.pop"][batch], state["algorithm.fit"][batch], state["algorithm.cv"][batch]
        )
    if name == "RVEAa":
        valid = torch.isfinite(state["algorithm.fit"]).all(2)
        assert (valid.sum(1) <= algorithm.pop_size).all()
