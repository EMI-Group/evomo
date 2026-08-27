import inspect
from unittest import TestCase

import torch

from evomo.problems.constrained import *  # noqa: F403


class TestConstrainedProblems(TestCase):
    def setUp(self):
        pass

    def _run_test_suite(self, problems):
        for prob_cls in problems:
            with self.subTest(problem=prob_cls.__name__):
                # Inspection of constructor signature to handle different parameters
                sig = inspect.signature(prob_cls.__init__)
                kwargs = {}
                if "m" in sig.parameters:
                    kwargs["m"] = 2
                if "d" in sig.parameters:
                    kwargs["d"] = 10

                # Suite-specific dimension adjustments
                if prob_cls.__name__.startswith("LSCM"):
                    kwargs["d"] = 100

                # M=3 cases
                m3_suites = ["DASCMOP7", "DASCMOP8", "DASCMOP9", "DOC8", "DOC9", "LIRCMOP13", "LIRCMOP14"]
                if any(prob_cls.__name__.startswith(s) for s in m3_suites):
                    kwargs["m"] = 3

                prob = prob_cls(**kwargs)

                # Evaluate
                x = torch.rand((2, prob.d))
                f, cv = prob.evaluate(x)

                # Assertions
                self.assertEqual(f.shape, (2, prob.m), f"Problem {prob_cls.__name__} fitness shape mismatch")
                self.assertEqual(cv.shape, (2, prob.n_iq + prob.n_eq), f"Problem {prob_cls.__name__} CV shape mismatch")

                # Pareto front
                pf = prob.pf()
                self.assertIsNotNone(pf, f"Problem {prob_cls.__name__} pf() returned None")
                self.assertEqual(pf.shape[1], prob.m, f"Problem {prob_cls.__name__} pf shape mismatch")

    def test_ctp(self):
        self._run_test_suite([CTP1, CTP2, CTP3, CTP4, CTP5, CTP6, CTP7, CTP8])  # noqa: F405

    def test_dascmop(self):
        self._run_test_suite([DASCMOP1, DASCMOP2, DASCMOP3, DASCMOP4, DASCMOP5, DASCMOP6, DASCMOP7, DASCMOP8, DASCMOP9])  # noqa: F405

    def test_doc(self):
        self._run_test_suite([DOC1, DOC2, DOC3, DOC4, DOC5, DOC6, DOC7, DOC8, DOC9])  # noqa: F405

    def test_fcp(self):
        self._run_test_suite([FCP1, FCP2, FCP3, FCP4, FCP5])  # noqa: F405

    def test_lircmop(self):
        self._run_test_suite(
            [
                LIRCMOP1,  # noqa: F405
                LIRCMOP2,  # noqa: F405
                LIRCMOP3,  # noqa: F405
                LIRCMOP4,  # noqa: F405
                LIRCMOP5,  # noqa: F405
                LIRCMOP6,  # noqa: F405
                LIRCMOP7,  # noqa: F405
                LIRCMOP8,  # noqa: F405
                LIRCMOP9,  # noqa: F405
                LIRCMOP10,  # noqa: F405
                LIRCMOP11,  # noqa: F405
                LIRCMOP12,  # noqa: F405
                LIRCMOP13,  # noqa: F405
                LIRCMOP14,  # noqa: F405
            ]
        )

    def test_lscm(self):
        self._run_test_suite([LSCM1, LSCM2, LSCM3, LSCM4, LSCM5, LSCM6, LSCM7, LSCM8, LSCM9, LSCM10, LSCM11, LSCM12])  # noqa: F405

    def test_mw(self):
        self._run_test_suite([MW1, MW2, MW3, MW4, MW5, MW6, MW7, MW8, MW9, MW10, MW11, MW12, MW13, MW14])  # noqa: F405
