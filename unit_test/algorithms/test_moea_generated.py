import unittest

import torch

from evomo.algorithms import (
    AGEMOEA,
    BCE_IBEA,
    BCEMOEAD,
    CLIA,
    CMOEA_MS,
    CMOPSO,
    DMMOEA,
    EFRRR,
    GDE3,
    GWASFGA,
    LSMOF,
    MOEAD_DCWV,
    MOEAD_DRA,
    MOEADAWA,
    MOEADDE,
    MOEADDU,
    MOEADDYTS,
    MOEADFRRMAB,
    MOEAURAW,
    NSGAII_SDR,
    OSP_NSDE,
    PESA2,
    PREA,
    SIBEA,
    SMPSO,
    SNSGA2,
    SPEAR,
    SSCEA,
    TELSO,
    TSNSGAII,
    WASFGA,
    WOF,
    BiGE,
    CoMMEA,
    GrEA,
    KnEA,
    MaOEACSS,
    MOEAD_PaS,
    NSBiDiCo,
    PICEAg,
    SparseEA,
    SparseEA2,
    ThetaDEA,
    TSSparseEA,
    Two_Arch2,
    VaEA,
    eMOEA,
    tDEA_CPBI,
)
from evomo.problems.numerical import DTLZ2
from evomo.workflows import UnifiedWorkflow

GENERATED_ALGORITHMS = (
    AGEMOEA,
    BCE_IBEA,
    BCEMOEAD,
    BiGE,
    CLIA,
    CMOEA_MS,
    CMOPSO,
    CoMMEA,
    DMMOEA,
    eMOEA,
    EFRRR,
    GDE3,
    GrEA,
    GWASFGA,
    KnEA,
    LSMOF,
    MaOEACSS,
    MOEADAWA,
    MOEAD_DCWV,
    MOEADDE,
    MOEAD_DRA,
    MOEADDU,
    MOEADDYTS,
    MOEADFRRMAB,
    MOEAD_PaS,
    MOEAURAW,
    NSBiDiCo,
    NSGAII_SDR,
    OSP_NSDE,
    PESA2,
    PICEAg,
    PREA,
    SNSGA2,
    SIBEA,
    SMPSO,
    SparseEA,
    SparseEA2,
    SPEAR,
    SSCEA,
    ThetaDEA,
    tDEA_CPBI,
    TELSO,
    TSNSGAII,
    TSSparseEA,
    Two_Arch2,
    VaEA,
    WASFGA,
    WOF,
)


class TestGeneratedAlgorithms(unittest.TestCase):
    def test_all_algorithms_run_dtlz2(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lb = torch.zeros(12, device=device)
        ub = torch.ones(12, device=device)

        for algorithm_cls in GENERATED_ALGORITHMS:
            with self.subTest(algorithm=algorithm_cls.__name__):
                torch.manual_seed(42)
                algorithm = algorithm_cls(pop_size=20, n_objs=3, lb=lb, ub=ub)
                workflow = UnifiedWorkflow(algorithm, DTLZ2(m=3), device=device)

                workflow.init_step()
                workflow.step()

                population = workflow.algorithm.pop
                fitness = workflow.algorithm.fit
                self.assertEqual(population.ndim, 2)
                self.assertEqual(fitness.ndim, 2)
                self.assertEqual(population.shape[0], fitness.shape[0])
                self.assertEqual(population.shape[1], 12)
                self.assertEqual(fitness.shape[1], 3)
                self.assertTrue(torch.isfinite(population).all().item())
                self.assertTrue(torch.isfinite(fitness).all().item())


if __name__ == "__main__":
    unittest.main()
