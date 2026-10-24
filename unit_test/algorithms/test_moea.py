import math
from unittest import TestCase

import torch
from evox.core import Algorithm, compile, use_state, vmap
from evox.workflows import EvalMonitor

from evomo.algorithms import IBEA, LMOCSO, MOEAD, NSGA2, NSGA3, RVEA, GrEA, HypE, RVEAa, TensorMOEAD
from evomo.problems.numerical import DTLZ2
from evomo.workflows import UnifiedWorkflow


class MOTestBase(TestCase):
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
            gr, gcd = g.sum(1), torch.zeros(n, dtype=fit.dtype)
            gcpd = torch.sqrt((((fit - (lb + g * d)) / d) ** 2).sum(1))
            gd = (g[:, None] - g[None]).abs().sum(-1).fill_diagonal_(math.inf)
            dom = (g[:, None] <= g[None]).all(-1) & (g[:, None] < g[None]).any(-1)
            remain = torch.ones(n, dtype=torch.bool)
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
                pd = torch.zeros(n, dtype=fit.dtype)
                for p in torch.where((gd[q] < m) & ngq & ~eq)[0].tolist():
                    if pd[p] < m - gd[q, p]:
                        pd[p] = m - gd[q, p]
                        for r in torch.where(dom[p] & remain & ~(gq | eq))[0].tolist():
                            pd[r] = max(pd[r], pd[p])
                gr[ngq & ~eq] += pd[ngq & ~eq]
            return ~remain

        generator = torch.Generator().manual_seed(0)
        for trial in range(30):
            n_objs = 2 + trial % 3
            n = 6 + trial
            fit = torch.rand(n, n_objs, generator=generator, dtype=torch.float64)
            if trial % 2 == 0:
                fit = torch.round(fit * 4) / 4  # shared cells and ties
            algo = GrEA(pop_size=n, n_objs=n_objs, lb=torch.zeros(2), ub=torch.ones(2), div=5 + trial % 6)
            k = n // 2
            chosen = algo._last_selection(fit.to(torch.get_default_device()), k).cpu()
            self.assertEqual(int(chosen.sum()), k)
            self.assertTrue(torch.equal(chosen, reference(fit, k, float(algo.div))))
