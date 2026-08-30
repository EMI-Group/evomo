import torch
from evox.core import Algorithm, Mutable
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import crowding_distance, tournament_selection_multifit
from evox.utils import lexsort

from evomo.operators.selection import non_dominate_rank
from evomo.utils import unique_rows_sorted


class SMPSO(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        D = lb.numel()

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        self.vel = Mutable(torch.zeros((pop_size, D), device=device))

        # Personal Best
        self.pbest_pop = Mutable(torch.zeros((pop_size, D), device=device))
        self.pbest_fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))

        # External Archive (Gbest)
        self.archive_pop = Mutable(torch.zeros((pop_size, D), device=device))
        self.archive_fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        self.archive_size = Mutable(torch.tensor(0, dtype=torch.int32, device=device))

    def _update_archive(self, off_pop, off_fit) -> None:
        device = self.lb.device
        # Combine current archive and new candidates
        valid_archive_mask = torch.arange(self.pop_size, device=device) < self.archive_size

        combined_pop = torch.cat([self.archive_pop[valid_archive_mask], off_pop], dim=0)
        combined_fit = torch.cat([self.archive_fit[valid_archive_mask], off_fit], dim=0)

        # Remove duplicates
        u_pop, u_idx = unique_rows_sorted(combined_pop)
        u_fit = combined_fit[u_idx]

        # Non-dominated sort
        rank = non_dominate_rank(u_fit)
        mask_rank1 = rank == 0
        rank1_fit = u_fit[mask_rank1]
        rank1_pop = u_pop[mask_rank1]

        num_rank1 = rank1_fit.shape[0]

        if num_rank1 > self.pop_size:
            # Truncate using crowding distance
            cd_mask = torch.ones(num_rank1, dtype=torch.bool, device=device)
            cd = crowding_distance(rank1_fit, cd_mask)
            # Sort by CD descending
            indices = lexsort(torch.stack([-cd]))
            selected_indices = indices[: self.pop_size]

            self.archive_pop = rank1_pop[selected_indices]
            self.archive_fit = rank1_fit[selected_indices]
            self.archive_size = torch.tensor(self.pop_size, dtype=torch.int32, device=device)
        else:
            # Fill archive and update size
            new_archive_pop = torch.zeros((self.pop_size, self.lb.numel()), device=device)
            new_archive_fit = torch.full((self.pop_size, self.n_objs), torch.inf, device=device)

            new_archive_pop[:num_rank1] = rank1_pop
            new_archive_fit[:num_rank1] = rank1_fit

            self.archive_pop = new_archive_pop
            self.archive_fit = new_archive_fit
            self.archive_size = torch.tensor(num_rank1, dtype=torch.int32, device=device)

    def init_step(self) -> None:
        self.fit = self.evaluate(self.pop)
        self.pbest_pop = self.pop.clone()
        self.pbest_fit = self.fit.clone()
        self._update_archive(self.pop, self.fit)

    def step(self) -> None:
        N = self.pop_size
        D = self.lb.numel()
        device = self.lb.device

        # 1. Leader Selection (Gbest)
        valid_mask = torch.arange(N, device=device) < self.archive_size
        # Extract valid archive members to compute CD safely
        valid_fits = self.archive_fit[valid_mask]
        num_valid = valid_fits.shape[0]

        # Initialize CD with a very small value for tournament selection
        cd = torch.full((N,), -1e6, device=device)

        # Only compute if archive is not empty
        if num_valid > 0:
            inner_mask = torch.ones(num_valid, dtype=torch.bool, device=device)
            cd_values = crowding_distance(valid_fits, inner_mask)
            # Use scatter to place values back into the N-sized tensor
            valid_indices = torch.arange(N, device=device)[valid_mask]
            cd.scatter_(0, valid_indices, cd_values)

        # Tournament selection to maximize CD (minimize -CD)
        # Invalid archive slots have -1e6 CD, so they won't be picked if valid ones exist
        adj_cd = -cd
        selected_indices = tournament_selection_multifit(N, [adj_cd], tournament_size=2)
        gbest_pop = self.archive_pop[selected_indices]

        # 2. Stochastic Parameters
        W = torch.rand((N, 1), device=device) * 0.4 + 0.1
        C1 = torch.rand((N, 1), device=device) * 1.0 + 1.5
        C2 = torch.rand((N, 1), device=device) * 1.0 + 1.5

        phi = torch.clamp(C1 + C2, min=4.0)
        chi = 2 / (torch.abs(2 - phi - torch.sqrt(phi**2 - 4 * phi)) + 1e-6)

        # 3. Velocity Update
        r1 = torch.rand((N, D), device=device)
        r2 = torch.rand((N, D), device=device)

        new_vel = chi * (W * self.vel + C1 * r1 * (self.pbest_pop - self.pop) + C2 * r2 * (gbest_pop - self.pop))

        # 4. Boundary & Deterministic Back
        delta = (self.ub - self.lb) / 2
        new_vel = torch.clamp(new_vel, -delta, delta)

        off_pop = self.pop + new_vel
        out_mask = (off_pop < self.lb) | (off_pop > self.ub)
        new_vel = torch.where(out_mask, new_vel * 0.001, new_vel)
        off_pop = torch.clamp(off_pop, self.lb, self.ub)

        # 5. Polynomial Mutation
        ind_mask = torch.rand(N, device=device) < 0.15
        dim_mask = torch.rand((N, D), device=device) < (1.0 / D)
        final_mutation_mask = ind_mask.unsqueeze(1) & dim_mask
        off_pop = polynomial_mutation(off_pop, self.lb, self.ub, final_mutation_mask)

        # 6. Evaluation
        off_fit = self.evaluate(off_pop)

        # 7. Pbest Update
        pbest_dom_off = (self.pbest_fit <= off_fit).all(dim=-1) & (self.pbest_fit < off_fit).any(dim=-1)
        replace_mask = ~pbest_dom_off

        self.pbest_pop = torch.where(replace_mask.unsqueeze(1), off_pop, self.pbest_pop)
        self.pbest_fit = torch.where(replace_mask.unsqueeze(1), off_fit, self.pbest_fit)

        # 8. Archive Update
        self._update_archive(off_pop, off_fit)

        # Update current swarm state
        self.pop = off_pop
        self.fit = off_fit
        self.vel = new_vel


if __name__ == "__main__":
    import time

    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = SMPSO(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
