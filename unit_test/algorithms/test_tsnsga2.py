import unittest

import torch

from evomo.algorithms import TSNSGAII
from evomo.operators.selection import non_dominate_rank


class TestTSNSGAIISelection(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.algo = TSNSGAII(pop_size=4, n_objs=2, lb=torch.zeros(3), ub=torch.ones(3))

    def test_spd_sort_peels_fronts(self):
        fit = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        rp = torch.tensor([0, 1, 2, 3])
        front_no = self.algo._spd_sort(fit, torch.zeros(4), rp, 4)
        self.assertEqual(front_no.tolist(), [1, 1, 2, 3])

    def test_spd_sort_breaks_ties_on_a_shared_reference_vector(self):
        # Mutually non-dominated, same reference vector: the smaller d1 + 5 * d2 comes first.
        fit = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        front_no = self.algo._spd_sort(fit, torch.tensor([0.3, 0.1]), torch.tensor([0, 0]), 2)
        self.assertEqual(front_no.tolist(), [2, 1])

    def test_decomposition_selection_keeps_non_dominated_solutions(self):
        # Five non-dominated solutions and three dominated ones: stage 2 picks pop_size = 4 non-dominated ones.
        t = torch.linspace(0, 1, 5)
        fit = torch.cat([torch.stack([t, 1 - t], dim=1), torch.tensor([[1.0, 1.0], [2.0, 2.0], [0.9, 0.9]])])
        pop = torch.arange(8.0).unsqueeze(1).repeat(1, 3)
        new_pop, new_fit, new_front, density = self.algo._decomposition_selection(pop, fit)
        self.assertEqual(new_pop.shape[0], 4)
        self.assertEqual(len(set(new_pop[:, 0].tolist())), 4)
        self.assertTrue((new_pop[:, 0] < 5).all())
        self.assertTrue((non_dominate_rank(new_fit) == 0).all())
        self.assertTrue(torch.equal(new_front, non_dominate_rank(new_fit)))
        self.assertEqual(density.shape, (4,))


if __name__ == "__main__":
    unittest.main()
