import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.operators.crossover import simulated_binary_half
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.utils import clamp


class MOEAD_DCWV(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, T: int = 20, p: float = -1.0, **kwargs):
        """Initialize the MOEAD_DCWV population and optimization state.

        :param pop_size: Required. Requested number of candidate solutions. Use a positive integer; algorithm-specific
            minimums and reference-vector sampling are described below.
        :type pop_size: int
        :param n_objs: Required. Number of objectives, matching the second dimension of the problem's objective tensor.
            Objectives are minimized; this library targets two or more objectives.
        :type n_objs: int
        :param lb: Required. Lower decision bounds of shape ``(D,)``, where ``D`` is the number of decision variables.
            Use floating-point bounds with the same shape, dtype and device as ``ub``, and ``lb <= ub`` elementwise.
        :type lb: torch.Tensor
        :param ub: Required. Upper decision bounds of shape ``(D,)``. The decision dimension is inferred from the
            bounds, rather than passed separately. Match ``lb`` in shape, dtype and device.
        :type ub: torch.Tensor
        :param T: Default: ``20``. Number of nearest weight vectors in each mating neighborhood, including the
            subproblem itself. Use a positive integer. Values above the sampled population size are capped to that size.
        :type T: int
        :param p: Default: ``-1.0``. Weight-adaptation switch. ``-1.0`` enables estimation of a transformation parameter
            from current normalized fitness on each step. Any other value disables that branch; the numeric value itself
            is not applied as a fixed transformation in the current implementation.
        :type p: float
        :param kwargs: Default: ``{}``. Extra keyword arguments are accepted for constructor compatibility but are not
            read by this implementation. In particular, passing ``device=...`` here does not move tensors; place both
            bounds on the intended device before construction.
        :type kwargs: dict

        .. note::

            Use this algorithm through a workflow that connects the problem's evaluation method. Call
            ``workflow.init_step()`` before ``workflow.step()`` or compiling the step. The current evaluation path
            expects an objective tensor of shape ``(B, n_objs)``. It does not consume a ``(fitness,
            constraint_violation)`` tuple.

            Reference-vector sampling can change the requested population size. Read ``self.pop.shape[0]`` for the actual size.

            Tensor allocation uses ``lb.device``. There is no explicit device parameter; prepare both bounds on the intended
            device.
        """
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.T = T
        self.p_param = Parameter(p)
        D = lb.numel()

        # Initialize Weights
        W0, _ = uniform_sampling(pop_size, n_objs)
        W0 = W0.to(device=device)
        self.pop_size = W0.shape[0]  # Adjust to sampling
        self.T = min(T, self.pop_size)

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(self.pop_size, D, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((self.pop_size, n_objs), torch.inf, device=device))
        self.W = Mutable(W0.clone())
        self.W0 = Mutable(W0)
        self.z_min = Mutable(torch.full((1, n_objs), torch.inf, device=device))

        # Neighborhood
        dist_matrix = torch.cdist(self.W, self.W)
        self.B = Mutable(torch.topk(dist_matrix, k=self.T, largest=False).indices)

    def _set_weight_transform(self, W: torch.Tensor, p: torch.Tensor, M: int) -> torch.Tensor:
        # Bug #12: Safe division
        mask = W < (1.0 / M)
        term1 = W * p * M
        term2 = 1 - (1 - W) * (1 - p) * M / (M - 1 + 1e-6)
        return torch.where(mask, term1, term2)

    def init_step(self) -> None:
        """Evaluate the initial population and initialize algorithm state.

        Invoke through ``workflow.init_step()`` before the first optimization step.

        :returns: ``None``; results are stored in algorithm state.
        """
        self.fit = self.evaluate(self.pop)
        self.z_min = torch.min(self.fit, dim=0, keepdim=True)[0]

    def step(self) -> None:
        """Advance optimization and update population and fitness state in place.

        Invoke through ``workflow.step()`` after initialization. The caller controls termination.

        :returns: ``None``; results are stored in algorithm state.
        """
        device = self.pop.device
        N = self.pop_size
        M = self.n_objs
        T = self.T

        # 1. Dynamic Weight Update (Sub-step 2.1)
        if self.p_param == -1.0:
            f_max = torch.max(self.fit, dim=0)[0]
            f_norm = (self.fit - self.z_min) / (f_max - self.z_min + 1e-6)

            normP = torch.norm(f_norm, p=2, dim=1, keepdim=True)
            CosineP = torch.sum(f_norm / (M**0.5), dim=1, keepdim=True) / (normP + 1e-6)
            dist = normP * torch.sqrt(torch.clamp(1 - CosineP**2, min=0.0) + 1e-6)

            idx = torch.argmin(dist)
            # Bug #41: No .item()
            p_val = (normP[idx] * CosineP[idx]) / (M**0.5 + 1e-6)

            self.W = self._set_weight_transform(self.W0, p_val, M)
            dist_matrix = torch.cdist(self.W, self.W)
            self.B = torch.topk(dist_matrix, k=T, largest=False).indices

        # 2. Variation (Sub-step 2.2)
        # Bug #14: Randint device
        mating_pool_indices = torch.randint(0, T, (N, 2), device=device)
        row_idx = torch.arange(N, device=device).unsqueeze(1)
        mating_pool = self.B[row_idx, mating_pool_indices]  # [N, 2]

        # OperatorGAhalf logic: concat parents for SBX
        parents = torch.cat([self.pop[mating_pool[:, 0]], self.pop[mating_pool[:, 1]]], dim=0)
        offspring = simulated_binary_half(parents)
        offspring = polynomial_mutation(offspring, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)

        off_fit = self.evaluate(offspring)

        # 3. Environmental Selection (Tchebycheff Update)
        self.z_min = torch.min(self.z_min, torch.min(off_fit, dim=0, keepdim=True)[0])
        z_max = torch.max(self.fit, dim=0, keepdim=True)[0]

        # Bug #29: Batch Vectorization
        fit_neighbors = self.fit[self.B]  # [N, T, M]
        W_neighbors = self.W[self.B]  # [N, T, M]

        # g_old calculation
        diff_old = torch.abs(fit_neighbors - self.z_min)
        g_old = torch.max(diff_old / (z_max - self.z_min + 1e-6) / (W_neighbors + 1e-6), dim=2).values  # [N, T]

        # g_new calculation
        off_fit_expanded = off_fit.unsqueeze(1)  # [N, 1, M]
        diff_new = torch.abs(off_fit_expanded - self.z_min)
        g_new = torch.max(diff_new / (z_max - self.z_min + 1e-6) / (W_neighbors + 1e-6), dim=2).values  # [N, T]

        # Update Masking
        replace_mask = g_new < g_old  # [N, T]

        # To handle sequential update emulation in batch:
        # We use the indices from B and the mask to update pop and fit.
        # Note: If multiple offspring update the same neighbor, the last one in the batch wins.
        flat_B = self.B.reshape(-1)
        flat_mask = replace_mask.reshape(-1)

        # Expand offspring to match [N, T, D] then flatten
        off_pop_expanded = offspring.unsqueeze(1).expand(-1, T, -1).reshape(-1, offspring.shape[1])
        off_fit_expanded_flat = off_fit.unsqueeze(1).expand(-1, T, -1).reshape(-1, M)

        # Apply updates
        update_indices = flat_B[flat_mask]
        self.pop[update_indices] = off_pop_expanded[flat_mask]
        self.fit[update_indices] = off_fit_expanded_flat[flat_mask]


if __name__ == "__main__":
    import time

    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda")

    algo = MOEAD_DCWV(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
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
