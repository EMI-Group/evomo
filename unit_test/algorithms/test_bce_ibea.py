import unittest

import torch

from evomo.algorithms.bce_ibea import _niche_radius, _pc_selection


class TestBCEIBEASelection(unittest.TestCase):
    def test_niche_radius_skips_the_point_itself(self):
        # Points 0, 1, 3, 6 on a line: the 3rd nearest other point is at distance 6, 5, 3 and 6.
        obj = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [6.0, 0.0]])
        self.assertAlmostEqual(_niche_radius(obj).item(), (6 + 5 + 3 + 6) / 4, places=5)

    def test_pc_selection_prunes_the_most_crowded_points(self):
        # Five mutually non-dominated points; two near-duplicate pairs and one isolated point.
        fit = torch.tensor([[0.0, 1.0], [0.01, 0.99], [0.5, 0.5], [0.99, 0.01], [1.0, 0.0]])
        pop = torch.arange(5.0).unsqueeze(1)
        new_pop, new_fit, n_nd = _pc_selection(pop, fit, 3)
        kept = set(new_pop.squeeze(1).long().tolist())
        self.assertEqual(n_nd.item(), 5)
        self.assertEqual(len(kept), 3)
        self.assertIn(2, kept)
        self.assertEqual(len(kept & {0, 1}), 1)
        self.assertEqual(len(kept & {3, 4}), 1)
        self.assertTrue(torch.equal(new_fit, fit[new_pop.squeeze(1).long()]))


if __name__ == "__main__":
    unittest.main()
