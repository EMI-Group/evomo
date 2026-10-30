import math

import torch
from evox.core import Algorithm, compile, use_state, vmap
from evox.workflows import EvalMonitor

from evomo.algorithms import (
    IBEA,
    LMOCSO,
    MOEAD,
    NSGA2,
    NSGA3,
    RVEA,
    SIBEA,
    GrEA,
    HypE,
    RVEAa,
    TensorMOEAD,
    VaEA,
)
from evomo.problems.numerical import DTLZ2
from evomo.workflows import UnifiedWorkflow
from unit_test import TorchTestCase


class MOTestBase(TorchTestCase):
    def run_algorithm(self, algo: Algorithm):
        prob = DTLZ2(m=3)
        monitor = EvalMonitor(multi_obj=True, full_sol_history=True)
        workflow = UnifiedWorkflow(algo, prob, monitor=monitor)
        workflow.init_step()
        for _ in range(3):
            workflow.step()

        monitor.get_pf()
        monitor.get_pf_fitness()

    def run_compiled_algorithm(self, algo: Algorithm):
        prob = DTLZ2(m=3)
        monitor = EvalMonitor(multi_obj=True, full_sol_history=True)
        workflow = UnifiedWorkflow(algo, prob, monitor=monitor)
        workflow.init_step()
        jit_step = compile(workflow.step, dynamic=False)
        for _ in range(3):
            jit_step()

        monitor.get_pf()
        monitor.get_pf_fitness()

    def run_vmap_algorithm(self, algo: Algorithm):
        prob = DTLZ2(m=3)
        workflow = UnifiedWorkflow(algo, prob)
        state_step = use_state(workflow.step)
        vmap_state_step = vmap(state_step, randomness="different")
        params, buffers = torch.func.stack_module_state([workflow] * 3)
        state = params | buffers
        vmap_state_step = compile(vmap_state_step)
        for _ in range(3):
            state = vmap_state_step(state)


class TestMOVariants(MOTestBase):
    def setUp(self):
        super().setUp()
        torch.compiler.reset()
        torch.manual_seed(42)
        torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
        self.pop_size = 20
        self.dim = 10
        self.lb = -torch.ones(self.dim)
        self.ub = torch.ones(self.dim)

    def test_nsga2(self):
        algo = NSGA2(pop_size=self.pop_size, n_objs=3, lb=self.lb, ub=self.ub)
        self.run_algorithm(algo)
        self.run_compiled_algorithm(algo)

    def test_nsga3(self):
        algo = NSGA3(pop_size=self.pop_size, n_objs=3, lb=self.lb, ub=self.ub)
        self.run_algorithm(algo)
        # self.run_compiled_algorithm(algo)

    def test_rvea(self):
        algo = RVEA(pop_size=self.pop_size, n_objs=3, lb=self.lb, ub=self.ub)
        self.run_algorithm(algo)
        self.run_compiled_algorithm(algo)
        self.run_vmap_algorithm(algo)

    def test_moead(self):
        algo = MOEAD(pop_size=self.pop_size, n_objs=3, lb=self.lb, ub=self.ub)
        self.run_algorithm(algo)
        self.run_compiled_algorithm(algo)

    # @skip("Torch 2.7 bug when running on non-AVX512 CPU: https://github.com/pytorch/pytorch/issues/152172")
    def test_hype(self):
        algo = HypE(pop_size=self.pop_size, n_objs=3, lb=self.lb, ub=self.ub)
        self.run_algorithm(algo)
        self.run_compiled_algorithm(algo)
        self.run_vmap_algorithm(algo)

    def test_rveaa(self):
        algo = RVEAa(pop_size=self.pop_size, n_objs=3, lb=self.lb, ub=self.ub)
        self.run_algorithm(algo)
        self.run_compiled_algorithm(algo)

    def test_tensormoead(self):
        algo = TensorMOEAD(pop_size=self.pop_size, n_objs=3, lb=self.lb, ub=self.ub)
        self.run_algorithm(algo)
        self.run_compiled_algorithm(algo)
        self.run_vmap_algorithm(algo)

    def test_ibea(self):
        algo = IBEA(pop_size=self.pop_size, n_objs=3, lb=self.lb, ub=self.ub)
        self.run_algorithm(algo)
        self.run_compiled_algorithm(algo)
        self.run_vmap_algorithm(algo)

    def test_lmocso(self):
        algo = LMOCSO(pop_size=self.pop_size, n_objs=3, lb=self.lb, ub=self.ub)
        self.run_algorithm(algo)
        self.run_compiled_algorithm(algo)
        self.run_vmap_algorithm(algo)

    def test_grea_default_div(self):
        # Grid divisions recommended for GrEA depend on the number of objectives
        for n_objs, div in [(2, 45), (3, 15), (5, 9), (10, 12), (12, 12)]:
            algo = GrEA(pop_size=self.pop_size, n_objs=n_objs, lb=self.lb, ub=self.ub)
            self.assertEqual(int(algo.div), div)

    def test_grea_last_selection_matches_reference(self):
        # Compare with a direct port of the sequential last-front selection of GrEA (PlatEMO's LastSelection)
        def reference(fit, k, div):
            n, m = fit.shape
            fmax, fmin = fit.max(0).values, fit.min(0).values
            lb = fmin - (fmax - fmin) / 2 / div
            d = (fmax - fmin) * (1 + 1 / div) / div
            g = torch.nan_to_num(torch.floor((fit - lb) / d), nan=0.0)
            gr, gcd = g.sum(1), torch.zeros(n, dtype=fit.dtype, device=fit.device)
            gcpd = torch.sqrt((((fit - (lb + g * d)) / d) ** 2).sum(1))
            gd = (g[:, None] - g[None]).abs().sum(-1).fill_diagonal_(math.inf)
            dom = (g[:, None] <= g[None]).all(-1) & (g[:, None] < g[None]).any(-1)
            remain = torch.ones(n, dtype=torch.bool, device=fit.device)
            while int(remain.sum()) > n - k:
                cand = torch.where(remain)[0]
                cand = cand[gr[cand] == gr[cand].min()]
                cand = cand[gcd[cand] == gcd[cand].min()]
                q = int(cand[gcpd[cand].argmin()])
                remain[q] = False
                gcd = gcd + torch.clamp(m - gd[q], min=0)
                eq, gq = (gd[q] == 0) & remain, dom[q] & remain
                ngq = remain & ~gq
                gr[eq] += m + 2
                gr[gq] += m
                pd = torch.zeros(n, dtype=fit.dtype, device=fit.device)
                for p in torch.where((gd[q] < m) & ngq & ~eq)[0].tolist():
                    if pd[p] < m - gd[q, p]:
                        pd[p] = m - gd[q, p]
                        for r in torch.where(dom[p] & remain & ~(gq | eq))[0].tolist():
                            pd[r] = max(pd[r], pd[p])
                gr[ngq & ~eq] += pd[ngq & ~eq]
            return ~remain

        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            # Generate identical CPU inputs for each device, independently of the global default device.
            generator = torch.Generator(device="cpu").manual_seed(0)
            for trial in range(30):
                with self.subTest(device=device, trial=trial):
                    n_objs = 2 + trial % 3
                    n = 6 + trial
                    fit = torch.rand(n, n_objs, generator=generator, dtype=torch.float64, device="cpu")
                    if trial % 2 == 0:
                        fit = torch.round(fit * 4) / 4  # shared cells and ties
                    algo = GrEA(
                        pop_size=n,
                        n_objs=n_objs,
                        lb=torch.zeros(2, device=device),
                        ub=torch.ones(2, device=device),
                        div=5 + trial % 6,
                    )
                    k = n // 2
                    chosen = algo._last_selection(fit.to(device), k)
                    self.assertEqual(chosen.device, algo.pop.device)
                    self.assertEqual(int(chosen.sum()), k)
                    self.assertTrue(torch.equal(chosen.cpu(), reference(fit, k, float(algo.div))))

    def test_vaea_worse_elimination(self):
        # Eight mutually non-dominated solutions. P and Q are about 2 degrees apart and P converges better
        # (smaller sum of normalized objectives). Maximum-vector-angle-first adds Q, and worse elimination
        # must then let P, the remaining solution closest in angle to the selected ones, replace Q.
        E1, R, U, P = [0.0, 1.0], [0.2, 0.85], [0.3, 0.75], [0.49, 0.515]
        Q, V, S, E2 = [0.51, 0.50], [0.7, 0.32], [0.8, 0.21], [1.0, 0.0]
        device = torch.get_default_device()
        algo = VaEA(pop_size=4, n_objs=2, lb=torch.zeros(3), ub=torch.ones(3))
        algo.pop = torch.rand(4, 3, device=device)
        algo.fit = torch.tensor([E1, R, U, P], device=device)
        offspring_fit = torch.tensor([Q, V, S, E2], device=device)
        algo.evaluate = lambda x: offspring_fit[: x.shape[0]]
        algo.step()
        survivors = {tuple(round(v, 3) for v in row) for row in algo.fit.tolist()}
        self.assertIn(tuple(P), survivors)
        self.assertNotIn(tuple(Q), survivors)
        self.assertIn(tuple(E1), survivors)
        self.assertIn(tuple(E2), survivors)

    def test_sibea_keeps_population_size(self):
        # The environmental selection must return pop_size survivors every generation, including
        # generations in which the first non-dominated front alone exceeds pop_size
        algo = SIBEA(pop_size=self.pop_size, n_objs=3, lb=torch.zeros(12), ub=torch.ones(12))
        workflow = UnifiedWorkflow(algo, DTLZ2(m=3))
        workflow.init_step()
        for _ in range(10):
            workflow.step()
            self.assertEqual(algo.pop.shape[0], self.pop_size)
            self.assertEqual(algo.fit.shape[0], self.pop_size)
