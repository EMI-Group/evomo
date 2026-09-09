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

    def test_maf12_r_nonsep_identity(self):
        """Regression test: _r_nonsep with a=1 must be the identity (WFG r_nonsep).

        The old implementation used ``a // 2`` as a *multiplier*, which zeroes the
        output for a=1. MAF12 applies it to the position variables, so the bug
        decoupled the objectives from the position variables entirely and all
        algorithms converged to the same degenerate front.
        """
        pro = MAF12(12, 3)
        y = torch.tensor([[0.7], [0.2], [0.99]])
        out = pro._r_nonsep(y, 1)
        assert torch.allclose(out.flatten(), y.flatten(), atol=1e-6), (
            f"a=1 must be identity, got {out}"
        )

    def test_maf12_r_nonsep_matches_inline_last_col(self):
        """_r_nonsep(a=L) must agree with the inline last_col formula in evaluate.

        evaluate() computes the aggregated distance column as
        ``(sum + 2*SUM) / (L/2) / (1 + 2L - 2*(L/2))``; the helper must be
        structurally consistent with it (they use the same normalization).
        """
        pro = MAF12(12, 3)
        L = 10
        torch.manual_seed(0)
        y = torch.rand(3, L)
        sum_ = y.sum(dim=1)
        sum_pairs_once = (y.unsqueeze(2) - y.unsqueeze(1)).abs().sum(dim=(1, 2)) * 0.5
        expected = (sum_ + 2 * sum_pairs_once) / (L / 2) / (1 + 2 * L - 2 * (L // 2))
        assert torch.allclose(pro._r_nonsep(y, L), expected, atol=1e-6)

    def test_maf12_objectives_depend_on_position_variables(self):
        """Regression test: position variables must influence MAF12 objectives."""
        pro = MAF12(12, 3)
        X = torch.full((4, 12), 0.3)
        X[:, :2] = 0.5
        f1 = pro.evaluate(X)
        X[:, :2] = 0.9
        f2 = pro.evaluate(X)
        assert not torch.allclose(f1, f2), "MAF12 objectives must depend on position variables"
