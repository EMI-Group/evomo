import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest

import torch
import torch.nn as nn

from evox.algorithms import NSGA2
from evox.utils import ParamsAndVector
from evox.workflows import EvalMonitor, StdWorkflow
from src.evox.problems import MoRobtrolPro


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.features = nn.Sequential(nn.Linear(8, 4), nn.Tanh(), nn.Linear(4, 2))

    def forward(self, x):
        x = self.features(x)
        return torch.tanh(x)


class TestProTest(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"

        self.seed = 1234
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        self.model = SimpleMLP().to(self.device)
        self.adapter = ParamsAndVector(dummy_model=self.model)

        self.POP_SIZE = 8
        self.OBJs = 2

        model_params = dict(self.model.named_parameters())
        self.pop_center = self.adapter.to_vector(model_params)
        self.lower_bound = torch.full_like(self.pop_center, -5)
        self.upper_bound = torch.full_like(self.pop_center, 5)

        self.algorithm = NSGA2(
            pop_size=self.POP_SIZE,
            n_objs=self.OBJs,
            lb=self.lower_bound,
            ub=self.upper_bound,
            device=self.device,
        )
        self.algorithm.setup()

        self.obs_norm = {"clip_val": 5.0, "std_min": 1e-6, "std_max": 1e6}

        self.problem = MoRobtrolPro(
            policy=self.model,
            env_name="mo_swimmer",
            max_episode_length=1000,
            num_episodes=3,
            pop_size=self.POP_SIZE,
            device=self.device,
            backend="generalized",
            num_obj=self.OBJs,
            observation_shape=8,
            obs_norm=self.obs_norm,
            useless=False,
        )

        self.monitor = EvalMonitor(device=self.device)
        self.monitor.setup()

        self.workflow = StdWorkflow(opt_direction="max")
        self.workflow.setup(
            algorithm=self.algorithm,
            problem=self.problem,
            solution_transform=self.adapter,
            monitor=self.monitor,
            device=self.device,
        )

    def test_mlp_forward(self):
        x = torch.randn(1, 8).to(self.device)
        output = self.model(x)
        self.assertEqual(output.shape, (1, 2), "MLP's shape is (1,2)")

    def test_algorithm_initialization(self):
        self.assertEqual(self.algorithm.pop_size, self.POP_SIZE, "Population size should be initialized correctly")
        self.assertEqual(self.algorithm.n_objs, self.OBJs, "The number of objective should be initialized correctly")

    def test_monitor_logging(self):
        self.workflow.step()
        self.monitor = self.workflow.get_submodule("monitor")
        fitness_history = self.monitor.get_fitness_history()

        self.assertTrue(len(fitness_history) > 0, "At least one fitness value should be recorded")


if __name__ == "__main__":
    unittest.main()
