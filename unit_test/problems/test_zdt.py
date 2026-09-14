from unittest import TestCase

import torch

from evomo.problems.numerical import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6

PROBLEM_TYPES = (ZDT1, ZDT2, ZDT3, ZDT4, ZDT6)


def reference_evaluate(problem_type, x):
    """Independent row-wise definitions of the five continuous ZDT problems."""
    tail = x[:, 1:]
    if problem_type is ZDT6:
        f1 = 1 - torch.exp(-4 * x[:, 0]) * torch.sin(6 * torch.pi * x[:, 0]) ** 6
        g = 1 + 9 * tail.mean(dim=1) ** 0.25
        h = 1 - (f1 / g) ** 2
    else:
        f1 = x[:, 0]
        if problem_type is ZDT4:
            g = 1 + 10 * tail.shape[1] + (tail**2 - 10 * torch.cos(4 * torch.pi * tail)).sum(dim=1)
        else:
            g = 1 + 9 * tail.mean(dim=1)

        if problem_type is ZDT2:
            h = 1 - (f1 / g) ** 2
        elif problem_type is ZDT3:
            h = 1 - torch.sqrt(f1 / g) - (f1 / g) * torch.sin(10 * torch.pi * f1)
        else:
            h = 1 - torch.sqrt(f1 / g)
    return torch.stack((f1, g * h), dim=1)


class TestZDT(TestCase):
    def setUp(self):
        device = torch.get_default_device()
        torch.set_default_device("cpu")
        self.addCleanup(torch.set_default_device, device)

    def test_matches_independent_formula_for_different_dimensions(self):
        for n in (2, 7, 10, 12):
            rows = [
                torch.zeros(n, dtype=torch.float64),
                torch.linspace(0.0, 1.0, n, dtype=torch.float64),
                torch.full((n,), 0.25, dtype=torch.float64),
                torch.linspace(1.0, 0.0, n, dtype=torch.float64),
            ]
            pop = torch.stack(rows)
            for problem_type in PROBLEM_TYPES:
                with self.subTest(problem=problem_type.__name__, n=n):
                    expected = reference_evaluate(problem_type, pop)
                    actual = problem_type(n=n).evaluate(pop)
                    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)

    def test_batched_evaluation_matches_individual_rows(self):
        for n in (3, 10, 17):
            generator = torch.Generator(device="cpu").manual_seed(100 + n)
            pop = torch.rand((5, n), dtype=torch.float64, generator=generator)
            for problem_type in PROBLEM_TYPES:
                with self.subTest(problem=problem_type.__name__, n=n):
                    problem = problem_type(n=n)
                    batched = problem.evaluate(pop)
                    individual = torch.cat([problem.evaluate(row.unsqueeze(0)) for row in pop], dim=0)
                    torch.testing.assert_close(batched, individual, rtol=0, atol=0)

    def test_known_values(self):
        for n in (2, 10, 13):
            zeros = torch.zeros((1, n), dtype=torch.float64)
            for problem_type in (ZDT1, ZDT2, ZDT3, ZDT4):
                with self.subTest(problem=problem_type.__name__, n=n, point="origin"):
                    torch.testing.assert_close(
                        problem_type(n=n).evaluate(zeros),
                        torch.tensor([[0.0, 1.0]], dtype=zeros.dtype),
                        rtol=0,
                        atol=0,
                    )

            zdt6 = ZDT6(n=n)
            torch.testing.assert_close(
                zdt6.evaluate(zeros), torch.tensor([[1.0, 0.0]], dtype=zeros.dtype), rtol=0, atol=0
            )
            tail_ones = torch.ones((1, n), dtype=torch.float64)
            tail_ones[:, 0] = 0
            torch.testing.assert_close(
                zdt6.evaluate(tail_ones),
                torch.tensor([[1.0, 9.9]], dtype=zeros.dtype),
                rtol=0,
                atol=1e-14,
            )

    def test_population_and_pareto_front_contract_without_mutation(self):
        n = 12
        pop = torch.rand(7, n)
        original_pop = pop.clone()
        for problem_type in PROBLEM_TYPES:
            with self.subTest(problem=problem_type.__name__):
                problem = problem_type(n=n)
                fit = problem.evaluate(pop)
                torch.testing.assert_close(pop, original_pop, rtol=0, atol=0)
                self.assertEqual(fit.shape, (7, 2))
                self.assertEqual(problem.pf().shape[1], 2)
