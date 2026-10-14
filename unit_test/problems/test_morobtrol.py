"""Basic Brax/Playground coverage; optional engines are skipped if not installed."""

import copy
import importlib.util
import unittest

import torch
from evox.algorithms import NSGA2
from evox.utils import ParamsAndVector
from evox.workflows import EvalMonitor, StdWorkflow

HAS_BRAX = all(importlib.util.find_spec(name) is not None for name in ("jax", "brax"))
HAS_PLAYGROUND = HAS_BRAX and importlib.util.find_spec("mujoco_playground") is not None


@unittest.skipUnless(HAS_BRAX, "Install the neuroevolution extra to test MoRobtrol")
class TestMoRobtrol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax

        from evomo.problems.neuroevolution import MoRobtrol

        cls.problem_class = MoRobtrol
        cls.device = "cuda" if torch.cuda.is_available() and any(d.platform == "gpu" for d in jax.devices()) else "cpu"

    def make_problem(self, name="mo_swimmer", engine="brax", **options):
        if engine == "playground":
            info = self.problem_class.playground_task_info(name, **options)
            observation, action = info["observation_size"], info["action_size"]
        else:
            from brax import envs

            env = envs.get_environment(name, **({"backend": options["backend"]} if "backend" in options else {}))
            observation, action = env.observation_size, env.action_size
        policy = torch.nn.Sequential(torch.nn.Linear(observation, action), torch.nn.Tanh()).to(self.device)
        with torch.no_grad():
            policy[0].weight.zero_()
            policy[0].bias.fill_(0.1)
        problem = self.problem_class(
            policy,
            name,
            max_episode_length=3,
            num_episodes=2,
            pop_size=2,
            seed=42,
            rotate_key=False,
            device=self.device,
            engine=engine,
            **options,
        )
        return problem, policy

    def assert_fitness(self, problem, expected_objectives):
        result = problem.evaluate({})
        self.assertEqual(result.shape, (2, expected_objectives))
        self.assertEqual(result.device, problem.device)
        self.assertTrue(torch.isfinite(result).all().item())
        torch.testing.assert_close(problem.evaluate({}), result)
        return result

    def test_brax_population(self):
        problem, policy = self.make_problem()
        self.assert_fitness(problem, 2)
        params = {name: value.detach().unsqueeze(0).repeat(2, *([1] * value.ndim)) for name, value in policy.named_parameters()}
        torch.testing.assert_close(problem.evaluate(params), problem.evaluate({}))

    def test_brax_workflow(self):
        problem, policy = self.make_problem()
        adapter = ParamsAndVector(dummy_model=policy)
        center = adapter.to_vector(dict(policy.named_parameters()))
        algorithm = NSGA2(
            pop_size=2, lb=torch.full_like(center, -1), ub=torch.full_like(center, 1), n_objs=2, device=self.device
        )
        workflow = StdWorkflow(
            algorithm,
            problem,
            monitor=EvalMonitor(device=self.device),
            opt_direction="max",
            solution_transform=adapter,
            device=self.device,
        )
        workflow.init_step()
        workflow.step()
        self.assertEqual(algorithm.fit.shape, (2, 2))
        self.assertTrue(torch.isfinite(algorithm.fit).all().item())

    @unittest.skipUnless(HAS_PLAYGROUND, "Install the playground extra")
    def test_playground_legacy(self):
        problem, _ = self.make_problem(engine="playground")
        self.assert_fitness(problem, 2)

    @unittest.skipUnless(HAS_PLAYGROUND, "Install the playground extra")
    def test_native_profiles(self):
        names = self.problem_class.available_playground_tasks()
        self.assertIn("CartpoleBalance", names)
        self.assertIn("mo_swimmer", names)
        for name, count in (("ReacherEasy", 2), ("CartpoleBalance", 3)):
            with self.subTest(task=name):
                problem, _ = self.make_problem(name, engine="playground")
                self.assertEqual(len(problem.objective_names), count)
                self.assertEqual(problem.objective_names[-1], "action_efficiency")
                result = self.assert_fitness(problem, count)
                expected = torch.full_like(result[:, -1], 3 / (1 + torch.tanh(torch.tensor(0.1)).item() ** 2))
                torch.testing.assert_close(result[:, -1], expected)

    @unittest.skipUnless(HAS_PLAYGROUND, "Install the playground extra")
    def test_custom_four_objectives(self):
        import jax.numpy as jnp

        objectives = {f"objective_{i}": lambda p, a, n, value=i: jnp.asarray(value, dtype=jnp.float32) for i in range(1, 5)}
        problem, _ = self.make_problem("CartpoleBalance", engine="playground", objectives=objectives)
        result = self.assert_fitness(problem, 4)
        expected = torch.tensor([[3.0, 6.0, 9.0, 12.0]], device=self.device).expand(2, -1)
        torch.testing.assert_close(result, expected)

    @unittest.skipUnless(HAS_PLAYGROUND, "Install the playground extra")
    def test_compiled_evaluation(self):
        problem, _ = self.make_problem("CartpoleBalance", engine="playground")
        problem.useless = False
        initial = copy.deepcopy(problem.state_dict())
        expected = problem.evaluate({})
        stats = problem.obs_param.clone()
        self.assertEqual(expected.dtype, torch.float32)
        self.assertEqual(stats.dtype, torch.float64)
        self.assertEqual(stats[0].item(), 12)
        problem.load_state_dict(initial)
        torch.testing.assert_close(torch.compile(problem.evaluate, fullgraph=True)({}), expected)
        torch.testing.assert_close(problem.obs_param, stats)


if __name__ == "__main__":
    unittest.main()
