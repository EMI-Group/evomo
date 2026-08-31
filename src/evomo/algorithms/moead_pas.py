import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.utils import clamp

from evomo.operators.selection import non_dominate_rank


class MOEAD_PaS(Algorithm):
    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        T: int | None = None,
        max_gen: int = 100,
        **kwargs,
    ):
        super().__init__()
        device = lb.device
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.dim = lb.numel()
        self.max_gen = Parameter(max_gen)

        # UniformPoint may adjust the requested population size.
        W, n_actual = uniform_sampling(pop_size, n_objs)
        self.pop_size = n_actual
        default_T = (n_actual + 9) // 10
        self.T = min(max(2, default_T if T is None else T), n_actual)
        self.nr = (self.T + 9) // 10
        self.W = Mutable(W.to(device))

        self.pop = Mutable(torch.rand(self.pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((self.pop_size, n_objs), torch.inf, device=device))

        distance = torch.cdist(self.W, self.W)
        self.B = Mutable(torch.topk(distance, self.T, largest=False, dim=1).indices)

        self.p = Mutable(torch.ones((self.pop_size,), device=device))
        self.z = Mutable(torch.full((1, n_objs), torch.inf, device=device))
        self.znad = Mutable(torch.full((1, n_objs), -torch.inf, device=device))
        self.gen = Mutable(torch.tensor(0, dtype=torch.int32, device=device))

    def _update_nadir(self, fit: torch.Tensor) -> torch.Tensor:
        rank = non_dominate_rank(fit)
        first_front = rank == 0
        safe_mask = first_front | ~first_front.any()
        masked_fit = torch.where(safe_mask.unsqueeze(1), fit, torch.full_like(fit, -torch.inf))
        return torch.max(masked_fit, dim=0, keepdim=True).values

    def _normalize(self, fit: torch.Tensor) -> torch.Tensor:
        denominator = torch.clamp(self.znad - self.z, min=1e-12)
        return torch.clamp((fit - self.z) / denominator, min=0.0)

    def _calc_scalar_func(self, normalized_fit: torch.Tensor, weights: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        values = normalized_fit / weights
        scale = torch.max(values, dim=-1).values.clamp_min(1e-30)

        # Rescaling before exponentiation avoids float32 overflow for p up to 10.
        finite_p = torch.where(torch.isinf(p), torch.ones_like(p), p)
        scaled_values = values / scale.unsqueeze(-1)
        g_p = scale * torch.sum(scaled_values.pow(finite_p.unsqueeze(-1)), dim=-1).pow(1.0 / finite_p)
        return torch.where(torch.isinf(p), scale, g_p)

    def _mating_and_candidates(self) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.pop.device
        N = self.pop_size
        T = self.T

        use_neighbours = torch.rand(N, device=device) < 0.9
        neighbour_order = torch.argsort(torch.rand((N, T), device=device), dim=1)
        shuffled_neighbours = torch.gather(self.B, 1, neighbour_order)
        global_candidates = torch.argsort(torch.rand((N, N), device=device), dim=1)

        local_padding = torch.zeros((N, N - T), dtype=torch.long, device=device)
        local_candidates = torch.cat([shuffled_neighbours, local_padding], dim=1)
        candidates = torch.where(use_neighbours.unsqueeze(1), local_candidates, global_candidates)

        positions = torch.arange(N, device=device).unsqueeze(0)
        valid = (~use_neighbours).unsqueeze(1) | (positions < T)
        return candidates, valid

    def _environmental_selection(
        self,
        candidates: torch.Tensor,
        valid: torch.Tensor,
        off_pop: torch.Tensor,
        off_fit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.pop.device
        N = self.pop_size

        candidate_fit = self.fit[candidates]
        candidate_weights = self.W[candidates]
        candidate_p = self.p[candidates]
        normalized_old = self._normalize(candidate_fit)
        normalized_new = self._normalize(off_fit).unsqueeze(1).expand(-1, N, -1)

        g_old = self._calc_scalar_func(normalized_old, candidate_weights, candidate_p)
        g_new = self._calc_scalar_func(normalized_new, candidate_weights, candidate_p)
        better = valid & (g_new < g_old)
        selected = better & (torch.cumsum(better.to(torch.int32), dim=1) <= self.nr)

        # Each target may receive proposals from multiple offspring. The sequential
        # MATLAB loop ultimately retains the proposal with the best scalar value;
        # resolving that reduction explicitly also avoids nondeterministic CUDA writes.
        target_ids = torch.arange(N, device=device).view(1, 1, N)
        targets = candidates.unsqueeze(-1) == target_ids
        proposal_scores = torch.where(
            targets & selected.unsqueeze(-1),
            g_new.unsqueeze(-1),
            torch.full((), torch.inf, device=device),
        ).amin(dim=1)
        best_score, best_offspring = torch.min(proposal_scores, dim=0)
        replace = torch.isfinite(best_score)

        new_pop = torch.where(replace.unsqueeze(1), off_pop[best_offspring], self.pop)
        new_fit = torch.where(replace.unsqueeze(1), off_fit[best_offspring], self.fit)
        return new_pop, new_fit

    def _adapt_norm(self) -> None:
        device = self.pop.device
        N = self.pop_size
        p_set = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, torch.inf], device=device)
        normalized_fit = self._normalize(self.fit)

        values = normalized_fit.view(1, 1, N, self.n_objs) / self.W.view(1, N, 1, self.n_objs)
        scale = torch.max(values, dim=-1).values.clamp_min(1e-30)
        scaled_values = values / scale.unsqueeze(-1)
        finite_p = torch.where(torch.isinf(p_set), torch.ones_like(p_set), p_set).view(-1, 1, 1)
        g_p = scale * torch.sum(scaled_values.pow(finite_p.unsqueeze(-1)), dim=-1).pow(1.0 / finite_p)
        g = torch.where(torch.isinf(p_set).view(-1, 1, 1), scale, g_p)

        best_individual = torch.argmin(g, dim=2)
        best_fit = normalized_fit[best_individual]
        fit_norm = torch.linalg.vector_norm(best_fit, dim=-1)
        weight_norm = torch.linalg.vector_norm(self.W, dim=-1).unsqueeze(0)
        cosine = torch.sum(best_fit * self.W.unsqueeze(0), dim=-1) / (fit_norm * weight_norm).clamp_min(1e-30)
        distance = torch.sqrt(torch.clamp(1.0 - cosine.square(), min=0.0)) * fit_norm
        best_p = p_set[torch.argmin(distance, dim=0)]

        progress = torch.clamp(self.gen.float() / self.max_gen.float(), max=1.0)
        adapt = torch.rand(N, device=device) >= progress
        self.p = torch.where(adapt, best_p, self.p)

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.z = torch.min(self.fit, dim=0, keepdim=True).values
        self.znad = self._update_nadir(self.fit)

    def step(self) -> None:
        self.gen = self.gen + 1
        candidates, valid = self._mating_and_candidates()
        off_pop = self.pop + 0.5 * (self.pop[candidates[:, 0]] - self.pop[candidates[:, 1]])
        off_pop = polynomial_mutation(off_pop, self.lb, self.ub)
        off_pop = clamp(off_pop, self.lb, self.ub)
        off_fit = self.evaluate(off_pop)

        self.pop, self.fit = self._environmental_selection(candidates, valid, off_pop, off_fit)
        self.z = torch.min(self.z, torch.min(self.fit, dim=0, keepdim=True).values)
        self.znad = self._update_nadir(self.fit)
        self._adapt_norm()


if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = MOEAD_PaS(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
    prob = DTLZ2(m=3)
    pf = prob.pf()
    workflow = StdWorkflow(algo, prob)
    workflow.init_step()
    jit_state_step = torch.compile(workflow.step)

    jit_state_step()

    torch.cuda.synchronize()
    exec_start = time.perf_counter()

    for i in range(1, 50):
        jit_state_step()

        if (i + 1) % 5 == 0:
            fit = workflow.algorithm.fit
            fit = fit[~torch.any(torch.isnan(fit), dim=1)]
            print(f"Gen {i + 1} IGD: {igd(fit, pf)}")

    torch.cuda.synchronize()
    exec_time = time.perf_counter() - exec_start
    print(f"Execution time for Gen 2-50 (49 steps): {exec_time:.4f}s (Avg: {exec_time / 49:.4f}s/gen)")
