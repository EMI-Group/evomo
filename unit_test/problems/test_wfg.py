from unittest import TestCase

import torch

from evomo.problems.numerical import WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9


class TestWFG(TestCase):
    def setUp(self):
        self.pro = [WFG1(), WFG2(), WFG3(), WFG4(), WFG5(), WFG6(), WFG7(), WFG8(), WFG9()]

    def test_wfg(self):
        pop = torch.rand(50, 12) * (2 * torch.arange(1, 13, dtype=torch.float32))
        original_pop = pop.clone()
        for pro in self.pro:
            fit = pro.evaluate(pop)
            assert (pop - original_pop).sum() == 0
            assert fit.size() == (50, 3)
            assert pro.lower.size(0) == pro.d
            assert pro.upper.size(0) == pro.d
            pf = pro.pf()
            assert pf.size(1) == 3

    def test_wfg2_wfg3_adjust_distance_variables_to_even(self):
        assert WFG2(d=13, m=3).d == 14
        assert WFG3(d=13, m=3).d == 14
