import unittest

import torch
import torch.nn as nn
from evox.algorithms import NSGA2
from evox.utils import ParamsAndVector
from evox.workflows import EvalMonitor, StdWorkflow

from evomo.problems.neuroevolution import MoRobtrol


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.features = nn.Sequential(nn.Linear(8, 4), nn.Tanh(), nn.Linear(4, 2))

    def forward(self, x):
        return torch.tanh(self.features(x))


def setup_workflow(model, pop_size, max_episode_length, num_episodes, device):
    adapter = ParamsAndVector(dummy_model=model)
    model_params = dict(model.named_parameters())
    pop_center = adapter.to_vector(model_params)
    lower_bound = torch.full_like(pop_center, -5)
    upper_bound = torch.full_like(pop_center, 5)

    problem = MoRobtrol(
        policy=model,
        env_name="mo_swimmer",
        max_episode_length=max_episode_length,
        num_episodes=num_episodes,
        pop_size=pop_size,
        device=device,
        num_obj=2,
        observation_shape=8,
        obs_norm=torch.tensor([5.0, 1e-6, 1e6], device=device),
    )

    algorithm = NSGA2(
        pop_size=pop_size, lb=lower_bound, ub=upper_bound, n_objs=2, device=device
    )
    monitor = EvalMonitor(device=device)

    workflow = StdWorkflow(
        algorithm=algorithm,
        problem=problem,
        monitor=monitor,
        opt_direction="max",
        solution_transform=adapter,
        device=device,
    )
    return workflow, adapter, monitor


def run_workflow(workflow, adapter, monitor, compiled=False, generations=3):
    step_function = torch.compile(workflow.step) if compiled else workflow.step
    for index in range(generations):
        step_function()


class TestMoRobtrolProblem(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)

    def test_morobtrol_problem(self):
        model = SimpleMLP().to(self.device)
        workflow, adapter, monitor = setup_workflow(model, 8, 100, 2, self.device)
        run_workflow(workflow, adapter, monitor, compiled=False)

    def test_compiled_morobtrol_problem(self):
        model = SimpleMLP().to(self.device)
        workflow, adapter, monitor = setup_workflow(model, 8, 100, 2, self.device)
        run_workflow(workflow, adapter, monitor, compiled=True)
