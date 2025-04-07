from unittest import TestCase

import torch
from evox.problems.numerical import LSMOP1, LSMOP2, LSMOP3, LSMOP4, LSMOP5, LSMOP6, LSMOP7, LSMOP8, LSMOP9


class TestLSMOP(TestCase):
    def setUp(self):
        self.pro = [LSMOP1(), LSMOP2(), LSMOP3(), LSMOP4(), LSMOP5(), LSMOP6(), LSMOP7(), LSMOP8(), LSMOP9()]

    def test_lsmop(self):
        pop = torch.rand(50, 300)
        original_pop = pop.clone()
        for pro in self.pro:
            fit = pro.evaluate(pop)
            assert (pop - original_pop).sum() == 0
            assert fit.size() == (50, 3)
            pf = pro.pf()
            assert pf.size(1) == 3
