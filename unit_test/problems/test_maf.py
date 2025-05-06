from unittest import TestCase

import torch
from evox.problems.numerical import (
    MAF1,
    MAF2,
    MAF3,
    MAF4,
    MAF5,
    MAF6,
    MAF7,
    MAF8,
    MAF9,
    MAF10,
    MAF11,
    MAF12,
    MAF13,
    MAF14,
    MAF15,
)


class TestMAF(TestCase):
    def setUp(self):
        d = 12
        m = 2
        self.pro = [
            MAF1(m + 9, m),  # MAF 1 is only defined for d = m + 9
            MAF2(m + 9, m),  # MAF 2 is only defined for d = m + 9
            MAF3(m + 9, m),  # MAF 3 is only defined for d = m + 9
            MAF4(m + 9, m),  # MAF 4 is only defined for d = m + 9
            MAF5(d, m),
            MAF6(m + 9, m),  # MAF 6 is only defined for d = m + 9
            MAF7(m + 19, m), # MAF 7 is only defined for d = m + 19
            MAF8(2, 3),      # MAF 8 is only defined for d = 2 and m >= 3
            MAF9(2, 3),      # MAF 9 is only defined for d = 2 and m >= 3
            MAF10(m + 9, m), # MAF 10 is only defined for d = m + 9
            MAF11(d, m), # MAF 11 is only defined for d = m + 9
            MAF12(m + 9, m), # MAF 12 is only defined for d = m + 9
            MAF13(d, 3),     # MAF 13 is only defined for m >= 3
            MAF14(m * 20, m),# MAF 14 is only defined for d = m * 20
            MAF15(m * 20, m),# MAF 15 is only defined for d = m * 20
        ]

    def test_maf(self):
        for pro in self.pro:
            pop = torch.rand(7, pro.d)
            print(f"pro: {pro}")

            fit = pro.evaluate(pop)
            print(f"fit.size(): {fit.size()}")
            assert fit.size() == (7, pro.m)

            pf = pro.pf()
            print(f"pf.size(): {pf.size()}")
            assert pf.size(1) == pro.m
