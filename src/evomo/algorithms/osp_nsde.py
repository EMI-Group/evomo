from functools import lru_cache

import torch
from evox.core import Algorithm, Mutable
from evox.operators.mutation import polynomial_mutation
from evox.utils import clamp, lexsort

from evomo.operators.selection.constraint_handling import (
    cat_violation,
    rank_with_constraints,
    take_violation,
    total_violation,
)
from evomo.utils import parse_evaluate, register_lazy_buffer


def _osp_tournament_minimum(values, active):
    indices = torch.arange(len(values), device=values.device)
    while len(indices) > 1:
        left, right = indices[::2], indices[1::2]
        if len(indices) % 2:
            right = torch.cat([right, left[-1:]])
        a, b = values[left], values[right]
        different = a != b
        column = different.to(torch.int32).argmax(1, keepdim=True)
        less = a.gather(1, column).squeeze(1) < b.gather(1, column).squeeze(1)
        less = less | (~different.any(1) & (left < right))
        choose_left = (active[left] & ~active[right]) | ((active[left] == active[right]) & less)
        indices = torch.where(choose_left, left, right)
    return indices[0]


def _osp_delete_one(values, neighbors, active):
    size = values.shape[0]
    row = torch.arange(size, device=values.device)
    column = torch.arange(size, device=values.device)
    if size > 128 and values.device.type != "cpu":
        remove = _osp_tournament_minimum(values, active)
    else:
        if not torch.compiler.is_compiling() and values.device.type == "cpu":
            nearest = values[:, 0].masked_fill(~active, torch.inf)
            indices = torch.where(active & (nearest == nearest.amin()))[0]
            ordered = values[indices]
        else:
            indices, ordered = row, values
        different = ordered[:, None] != ordered[None]
        first = torch.where(different, column, size).amin(2)
        unequal = first < size
        first = first.clamp_max(size - 1)
        left = ordered[:, None].expand(-1, len(indices), -1).gather(2, first[:, :, None]).squeeze(2)
        right = ordered[None].expand(len(indices), -1, -1).gather(2, first[:, :, None]).squeeze(2)
        earlier = (unequal & (left < right)) | (~unequal & (indices[:, None] < indices[None]))
        preceded = (earlier & active[indices, None]).any(0)
        remove = torch.where(active[indices] & ~preceded, indices, size).amin()
    position = (neighbors == remove).to(torch.int32).argmax(1)
    source = (column[None] + (column[None] >= position[:, None])).clamp_max(size - 1)
    values = values.gather(1, source).masked_fill(column[None] == size - 1, torch.inf)
    neighbors = neighbors.gather(1, source).masked_fill(column[None] == size - 1, size)
    active = active.scatter(0, remove.reshape(1), False)
    return values, neighbors, active


def _osp_truncate_mask(distance, active, keep):
    """Delete one row at a time, sorting its neighbor distances only once.

    ``active`` identifies the cutoff front in a padded, fixed-shape population.
    Complete better fronts are handled by the caller. Original indices break ties.
    """
    size = distance.shape[0]
    diagonal = torch.eye(size, dtype=torch.bool, device=distance.device)
    distance = distance.masked_fill(diagonal | ~active[None], torch.inf)
    values, neighbors = distance.sort(1)
    remaining = active.sum() - keep
    if torch.compiler.is_compiling():

        def cond(v, n, a, r):
            return r > 0

        def body(v, n, a, r):
            v, n, a = _osp_delete_one(v, n, a)
            return v, n, a, r - 1

        _, _, active, _ = torch.while_loop(cond, body, (values, neighbors, active, remaining))
    else:
        for _ in range(int(remaining)):
            values, neighbors, active = _osp_delete_one(values, neighbors, active)
    return active


def _osp_truncate_cpu_front(distance, count):
    """Use compact comparisons for short CPU fronts, where row-cache updates cost more."""
    size = len(distance)
    active = torch.ones(size, dtype=torch.bool, device=distance.device)
    distance = distance.masked_fill(torch.eye(size, dtype=torch.bool, device=distance.device), torch.inf)
    for _ in range(size - count):
        current = distance.masked_fill(~active[None], torch.inf)
        nearest = current.amin(1).masked_fill(~active, torch.inf)
        # A lexicographic minimum must first minimize the nearest distance.
        # Restricting the full comparison to these ties avoids N tiny CPU sorts.
        indices = torch.where(active & (nearest == nearest.amin()))[0]
        ordered = current[indices].sort(1).values
        unequal = ordered[:, None] != ordered[None]
        first = unequal.to(torch.int32).argmax(2)
        left = ordered[:, None].expand(-1, len(indices), -1).gather(2, first[:, :, None]).squeeze(2)
        right = ordered[None].expand(len(indices), -1, -1).gather(2, first[:, :, None]).squeeze(2)
        different = unequal.any(2)
        earlier = (different & (left < right)) | (~different & (indices[:, None] < indices[None]))
        remove = indices[(~earlier.any(0)).to(torch.int32).argmax()]
        active[remove] = False
    return active


def _hypervolume_2d(points, reference):
    order = points[:, 0].argsort(stable=True)
    values = points[order]
    width = torch.cat([values[1:, 0], reference[:1]]) - values[:, 0]
    height = reference[1] - values[:, 1].cummin(0).values
    return (width.clamp_min(0) * height.clamp_min(0)).sum()


def _hypervolume(points, reference, samples=None, *, mask=None):
    if not len(points):
        return reference.new_zeros(())
    valid = (points < reference).all(1) & points.isfinite().all(1)
    if mask is not None:
        valid = valid & mask
    points = torch.where(valid[:, None], points, reference[None])
    if points.shape[1] == 2:
        return _hypervolume_2d(points, reference)
    if points.shape[1] == 3:
        x = points[:, 0].sort().values
        ordered = points[points[:, 1].argsort(stable=True)]
        eligible = ordered[None, :, 0] <= x[:, None]
        lowest = torch.where(eligible, ordered[None, :, 2], torch.inf).cummin(1).values
        dy = torch.cat([ordered[1:, 1], reference[1:2]]) - ordered[:, 1]
        area = (dy[None].clamp_min(0) * (reference[2] - lowest).clamp_min(0)).sum(1)
        dx = torch.cat([x[1:], reference[:1]]) - x
        return (dx.clamp_min(0) * area).sum()
    lower = points.amin(0)
    locations = lower + samples * (reference - lower)
    dominated = (points[:, None] <= locations[None]).all(2).any(0)
    return dominated.to(points.dtype).mean() * (reference - lower).prod()


def _forecast_ar1(history, horizon):
    """Univariate ARX(1) with no exogenous input, fitted over time rather than population rows."""
    if history.shape[-1] < 3:
        return history[..., -1]
    previous, following = history[..., :-1], history[..., 1:]
    denominator = previous.square().sum(-1)
    coefficient = (previous * following).sum(-1) / denominator.clamp_min(torch.finfo(history.dtype).eps)
    coefficient = torch.where(denominator > 0, coefficient, 1).clamp(-2, 2)
    return history[..., -1] * coefficient.pow(int(horizon))


def _variational_gaussian_mixture(values, components, max_iterations=200, tolerance=1e-8):
    """Normal-Wishart/Dirichlet variational mixture with the reference's prior strengths."""
    n, d = values.shape
    k = min(components, n)
    dtype, device = values.dtype, values.device
    identity = torch.eye(d, dtype=dtype, device=device)
    prior_mean = values.mean(0)
    assignment = torch.randint(k, (n,), device=device)
    responsibility = torch.nn.functional.one_hot(assignment, k).to(dtype)
    previous = responsibility
    for iteration in range(max_iterations):
        count = responsibility.sum(0)
        kappa = count + 0.001
        alpha = count + 0.1
        nu = count + d
        weighted = responsibility.T @ values
        means = (weighted + 0.001 * prior_mean) / kappa[:, None]
        centered = values[:, None] - means[None]
        scatter = torch.einsum("nk,nki,nkj->kij", responsibility, centered, centered)
        difference = prior_mean - means
        inverse_w = 1e-5 * identity[None] + scatter + 0.001 * difference[:, :, None] * difference[:, None, :]
        chol = torch.linalg.cholesky(inverse_w + torch.finfo(dtype).eps * identity[None])
        precision = torch.cholesky_inverse(chol)
        quadratic = d / kappa[None] + nu[None] * torch.einsum("nki,kij,nkj->nk", centered, precision, centered)
        log_det_w = -2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(1)
        dims = torch.arange(d, dtype=dtype, device=device)
        log_precision = torch.digamma((nu[:, None] - dims[None]) / 2).sum(1) + d * torch.log(values.new_tensor(2)) + log_det_w
        log_weight = torch.digamma(alpha) - torch.digamma(alpha.sum())
        responsibility = torch.softmax(log_weight + 0.5 * log_precision - 0.5 * quadratic, dim=1)
        if iteration > 1 and bool((responsibility - previous).abs().amax() < tolerance):
            break
        previous = responsibility
    # Posterior predictive covariance; positive regularization also covers singleton components.
    covariance = inverse_w * ((kappa + 1) / (kappa * (nu - d + 1).clamp_min(1)))[:, None, None]
    weights = responsibility.sum(0).clamp_min(torch.finfo(dtype).eps)
    return means, covariance + 1e-6 * identity[None], weights / weights.sum()


def _sample_mixture(means, covariance, weights, count):
    components = torch.multinomial(weights, count, replacement=True)
    chol = torch.linalg.cholesky(covariance)
    noise = torch.randn(count, means.shape[1], dtype=means.dtype, device=means.device)
    return means[components] + torch.einsum("nij,nj->ni", chol[components], noise)


@lru_cache(maxsize=1)
def _get_cuda_truncation():
    """Build and register optional CUDA kernels only when an OSP instance needs them."""
    import triton
    import triton.language as tl

    @triton.jit
    def trunc_kernel(V, Neighbors, A, K, OUT, N: tl.constexpr, B: tl.constexpr):
        row = tl.arange(0, B)
        col = tl.arange(0, B)
        values = tl.load(V + row[:, None] * N + col[None], (row[:, None] < N) & (col[None] < N), other=float("inf"))
        ids = tl.load(Neighbors + row[:, None] * N + col[None], (row[:, None] < N) & (col[None] < N), other=N)
        active = tl.load(A + row, row < N, other=0).to(tl.int1)
        # Keep sorted rows on the GPU and compact them after each exact deletion.
        drops = tl.sum(active.to(tl.int32), 0) - tl.load(K)
        while drops > 0:
            # Lexicographic comparison stops at the first unequal distance.
            candidates = active
            column = 0
            ties = tl.sum(candidates.to(tl.int32), 0)
            while (ties > 1) & (column < N):
                v = tl.gather(values, tl.full((B, 1), column, tl.int32), 1).reshape(B)
                minimum = tl.min(tl.where(candidates, v, float("inf")), 0)
                candidates = candidates & (v == minimum)
                column += 1
                ties = tl.sum(candidates.to(tl.int32), 0)
                if (column == 2) & (ties > 1):
                    first = tl.min(tl.where(candidates, row, N), 0)
                    reference = tl.sum(tl.where(row[:, None] == first, values, 0.0), 0)
                    different = tl.sum(((values != reference[None]) & candidates[:, None]).to(tl.int32), 0)
                    if tl.sum(different, 0) == 0:
                        column = N
            remove = tl.min(tl.where(candidates, row, N), 0)
            position = tl.max(tl.where(ids == remove, col[None], 0), 1)
            source = tl.minimum(col[None] + (col[None] >= position[:, None]).to(tl.int32), B - 1)
            values = tl.gather(values, source, 1)
            ids = tl.gather(ids, source, 1)
            values = tl.where(col[None] < B - 1, values, float("inf"))
            ids = tl.where(col[None] < B - 1, ids, N)
            active = active & (row != remove)
            drops -= 1
        tl.store(OUT + row, active, row < N)

    @triton.jit
    def _streamed_kernel(V, Neighbors, A, K, OUT, N: tl.constexpr, B: tl.constexpr, R: tl.constexpr):
        row = tl.arange(0, B)
        col = tl.arange(0, B)
        active = tl.load(A + row, row < N, other=0).to(tl.int1)
        drops = tl.sum(active.to(tl.int32), 0) - tl.load(K)
        while drops > 0:
            candidates = active
            column = 0
            ties = tl.sum(candidates.to(tl.int32), 0)
            while (ties > 1) & (column < N):
                nearest_values = tl.load(V + row * N + column, candidates & (row < N), other=float("inf"))
                minimum = tl.min(nearest_values, 0)
                candidates = candidates & (nearest_values == minimum)
                column += 1
                ties = tl.sum(candidates.to(tl.int32), 0)
                if (column == 2) & (ties > 1):
                    first = tl.min(tl.where(candidates, row, N), 0)
                    reference = tl.load(V + first * N + col, col < N, other=float("inf"))
                    mismatch = tl.full((), 0, tl.int32)
                    for start in range(0, N, R):
                        rows = start + tl.arange(0, R)
                        selected = tl.gather(candidates, tl.minimum(rows, B - 1), 0) & (rows < N)
                        values = tl.load(
                            V + rows[:, None] * N + col[None], selected[:, None] & (col[None] < N), other=float("inf")
                        )
                        different = (values != reference[None]) & selected[:, None]
                        mismatch += tl.sum(tl.sum(different.to(tl.int32), 1), 0)
                    if mismatch == 0:
                        column = N
            remove = tl.min(tl.where(candidates, row, N), 0)
            for start in range(0, N, R):
                rows = start + tl.arange(0, R)
                valid = (rows[:, None] < N) & (col[None] < N)
                ids = tl.load(Neighbors + rows[:, None] * N + col[None], valid, other=N).to(tl.int32)
                values = tl.load(V + rows[:, None] * N + col[None], valid, other=float("inf"))
                position = tl.max(tl.where(ids == remove, col[None], 0), 1)
                source = tl.minimum(col[None] + (col[None] >= position[:, None]).to(tl.int32), B - 1)
                values = tl.gather(values, source, 1)
                ids = tl.gather(ids, source, 1)
                values = tl.where(col[None] < B - 1, values, float("inf"))
                ids = tl.where(col[None] < B - 1, ids, N)
                tl.store(V + rows[:, None] * N + col[None], values, valid)
                tl.store(Neighbors + rows[:, None] * N + col[None], ids, valid)
            active = active & (row != remove)
            drops -= 1
        tl.store(OUT + row, active, row < N)

    @torch.library.triton_op("evomo_osp::truncate", mutates_args={})
    def _cuda_truncate_mask(distance: torch.Tensor, active: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        size = distance.shape[0]
        diagonal = torch.eye(size, dtype=torch.bool, device=distance.device)
        values, indices = distance.masked_fill(diagonal | ~active[None], torch.inf).sort(1)
        result = torch.empty_like(active)
        if size <= 128:
            torch.library.wrap_triton(trunc_kernel)[(1,)](
                values, indices, active, keep, result, size, triton.next_power_of_2(size), num_warps=8, enable_fp_fusion=False
            )
        else:
            torch.library.wrap_triton(_streamed_kernel)[(1,)](
                values,
                indices,
                active,
                keep,
                result,
                size,
                triton.next_power_of_2(size),
                8,
                num_warps=8,
                enable_fp_fusion=False,
            )
        return result

    return _cuda_truncate_mask


class OSP_NSDE(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, t_max: int = 100, **kwargs):
        """Initialize the OSP_NSDE population and optimization state.

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
        :param t_max: Default: ``100``. Initial trajectory-buffer capacity, including the initial entry.
            Buffers grow when needed; larger values reserve more history storage initially.
        :type t_max: int
        :param kwargs: Optional ``max_fe`` bounds actual evaluations, including probes and initialization.
            Place bounds on the intended device; ``device=...`` does not move them.
            ``lambda_`` (default ``0.2``), ``p`` (``50``), and ``local_search_steps`` (``8``) configure prediction.
            ``truncation_backend`` is ``"auto"`` by default: use the optional Triton cutoff-front kernel on
            CUDA float32/float64 with at most 128 individuals when available, and PyTorch otherwise.
            Set ``"torch"`` or ``"triton"`` explicitly to choose it. Non-dominated ranking stays in PyTorch.
        :type kwargs: dict

        .. note::

            Use this algorithm through a workflow that connects the problem's evaluation method. Call
            ``workflow.init_step()`` before ``workflow.step()`` or compiling the step. The current evaluation path
            accepts an objective tensor or a ``(fitness, constraint_violation)`` tuple.
            Constrained evaluations retain violations through selection and state updates.

            Tensor allocation uses ``lb.device``. There is no explicit device parameter; prepare both bounds on the intended
            device.
        """
        super().__init__()
        device = lb.device
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.lb = lb
        self.ub = ub
        self.dim = lb.numel()
        self.t_max = t_max

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, self.dim, dtype=lb.dtype, device=device) * (ub - lb) + lb)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, dtype=lb.dtype, device=device))

        # Trajectory Tracking
        self.history_pop = Mutable(torch.zeros((pop_size, self.dim, t_max), dtype=lb.dtype, device=device))
        self.history_fit = Mutable(torch.zeros((pop_size, n_objs, t_max), dtype=lb.dtype, device=device))
        self.history_hv = Mutable(torch.zeros((t_max,), dtype=lb.dtype, device=device))

        # Counters and Parameters
        self.t = Mutable(torch.tensor(0, dtype=torch.int32, device=device))
        self.t_init = Mutable(torch.tensor(2, dtype=torch.int64, device=device))
        self.p = Mutable(torch.tensor(kwargs.get("p", 50), dtype=torch.int64, device=device))
        self.nadir = Mutable(torch.ones(n_objs, dtype=lb.dtype, device=device) * 1.5)

        register_lazy_buffer(self, "cv", device_like="pop")
        self.max_fe = kwargs.get("max_fe")
        self.lambda_ = kwargs.get("lambda_", 0.2)
        self.local_search_steps = kwargs.get("local_search_steps", 8)
        self.fe = Mutable(torch.tensor(0, dtype=torch.int64, device=device))
        self.history_rank = Mutable(torch.full((pop_size, t_max), pop_size, dtype=torch.int32, device=device))
        samples = torch.quasirandom.SobolEngine(n_objs).draw(2048).to(device=device, dtype=lb.dtype)
        self.hv_samples = samples
        self.truncation_backend = kwargs.get("truncation_backend", "auto")
        if self.truncation_backend not in {"auto", "torch", "triton"}:
            raise ValueError("truncation_backend must be 'auto', 'torch' or 'triton'")
        self._truncate_operator = _osp_truncate_mask
        use_cuda_kernel = (
            device.type == "cuda"
            and 2 * pop_size <= 256
            and lb.dtype in (torch.float32, torch.float64)
            and torch.cuda.get_device_capability(device)[0] >= 8
            and all(hasattr(torch.library, name) for name in ("triton_op", "wrap_triton"))
        )
        if self.truncation_backend == "triton" and not use_cuda_kernel:
            raise ValueError("The OSP Triton kernel requires CUDA float32/float64 and pop_size <= 128")
        if use_cuda_kernel and self.truncation_backend != "torch":
            try:
                cuda_truncate = _get_cuda_truncation()
            except ImportError:
                if self.truncation_backend == "triton":
                    raise
            else:
                self._truncate_operator = cuda_truncate
        self.truncation_backend = "torch" if self._truncate_operator is _osp_truncate_mask else "triton"

    def _remaining(self):
        return 2**31 if self.max_fe is None else max(0, self.max_fe - int(self.fe))

    def _evaluate(self, pop):
        result = parse_evaluate(self.evaluate(pop))
        self.fe = self.fe + len(pop)
        return result

    def init_step(self) -> None:
        self.fit, self.cv = self._evaluate(self.pop)
        self.nadir = self.fit.amax(0).clamp_min(torch.finfo(self.fit.dtype).eps)
        self.history_pop[:, :, 0] = self.pop
        self.history_fit[:, :, 0] = self.fit
        self.history_rank[:, 0] = rank_with_constraints(self.fit, self.cv)

    @torch.compiler.disable
    def _cpu_selection_indices(self, fit, cv):
        """Keep short CPU front scans eager while the surrounding workflow compiles."""
        rank = rank_with_constraints(fit, cv)
        cutoff = rank.kthvalue(self.pop_size).values
        complete = torch.where(rank < cutoff)[0]
        last = torch.where(rank == cutoff)[0]
        values = fit[last]
        distance = (values[:, None] - values[None]).square().sum(2).sqrt()
        keep = self._truncate_last_front(distance, self.pop_size - len(complete))
        return torch.cat([complete, last[keep]]).sort().values

    def _selection_indices(self, fit, cv):
        if fit.device.type == "cpu":
            return self._cpu_selection_indices(fit, cv)
        rank = rank_with_constraints(fit, cv)
        cutoff = rank.kthvalue(self.pop_size).values
        complete, last = rank < cutoff, rank == cutoff
        # Direct differences are symmetric and keep duplicate distances exactly zero.
        distance = (fit[:, None] - fit[None]).square().sum(2).sqrt()
        keep = self.pop_size - complete.sum()
        active = self._truncate_operator(distance, last, keep)
        indices = torch.arange(len(fit), device=fit.device)
        return torch.where(complete | active, indices, len(fit)).sort().values[: self.pop_size]

    def _select(self, pop, fit, cv):
        indices = self._selection_indices(fit, cv)
        self.pop, self.fit, self.cv = pop[indices], fit[indices], take_violation(cv, indices)

    @staticmethod
    def _truncate_last_front(distance, count):
        if distance.device.type == "cpu":
            return _osp_truncate_cpu_front(distance, count)
        active = torch.ones(len(distance), dtype=torch.bool, device=distance.device)
        return _osp_truncate_mask(distance, active, distance.new_tensor(count, dtype=torch.int64))

    @torch.compiler.disable
    def _prepare_step(self):
        count = min(self.pop_size, self._remaining())
        if not count:
            return 0
        time = int(self.t) + 1
        if time >= self.history_hv.shape[0]:
            self.history_pop = torch.cat([self.history_pop, torch.zeros_like(self.history_pop)], 2)
            self.history_fit = torch.cat([self.history_fit, torch.zeros_like(self.history_fit)], 2)
            self.history_rank = torch.cat([self.history_rank, torch.full_like(self.history_rank, self.pop_size)], 1)
            self.history_hv = torch.cat([self.history_hv, torch.zeros_like(self.history_hv)])
        return count

    def _record_history(self, rank):
        time = self.t.long().reshape(1)
        self.history_pop.index_copy_(2, time, self.pop[:, :, None])
        self.history_fit.index_copy_(2, time, self.fit[:, :, None])
        self.history_rank.index_copy_(1, time, rank[:, None])
        self.nadir = torch.where(self.t == 2, self.fit.amax(0).clamp_min(torch.finfo(self.fit.dtype).eps), self.nadir)
        hv = _hypervolume(self.fit, 1.1 * self.nadir, self.hv_samples, mask=rank == 0)
        self.history_hv.index_copy_(0, time, hv.reshape(1))
        start = self.t_init.long().clamp_max(self.history_hv.shape[0] - 1).reshape(1)
        gain = hv >= (1 + self.lambda_) * self.history_hv.index_select(0, start).squeeze(0)
        trigger = (self.t - self.t_init >= 3) & ((self.t == 10) | (gain & ((rank == 0).sum() < 0.9 * self.pop_size)))
        if self.cv is not None:
            spread = (self.fit.amax(0) - self.fit.amin(0)).abs().sum() + total_violation(self.cv).std(unbiased=False)
            trigger = trigger & (spread > torch.finfo(self.fit.dtype).eps)
        return trigger

    @staticmethod
    @torch.compiler.disable
    def _prediction_due(trigger):
        return bool(trigger)

    def _arx_forecast(self, history, p):
        return _forecast_ar1(history, p)

    def _forecast_front(self, rank):
        indices = torch.where(rank == 0)[0]
        indices = indices[self.fit[indices, 0].argsort(stable=True)]
        series = []
        for time in range(max(0, int(self.t_init)), int(self.t) + 1):
            old = torch.where(self.history_rank[:, time] == 0)[0]
            if not len(old):
                old = indices
            values = self.history_fit[old, :, time]
            if len(old) == len(indices):
                aligned = values[values[:, 0].argsort(stable=True)]
            else:
                aligned = values[torch.cdist(self.fit[indices], values).argmin(1)]
            series.append(aligned)
        history = torch.stack(series, dim=-1)
        improving = (history[..., -1] < history[..., 0]).all(1)
        if bool(improving.any()):
            movement = (history[..., -1] - history[..., 0]).norm(dim=1)
            best = torch.where(improving, movement, -torch.inf).argmax()
            model = history[best : best + 1]
            previous, following = model[..., :-1], model[..., 1:]
            coefficient = (previous * following).sum(-1) / previous.square().sum(-1).clamp_min(1e-12)
            forecast = history[..., -1] * coefficient.clamp(-2, 2).pow(int(self.p))
        else:
            forecast = self._arx_forecast(history, self.p)
        return indices, forecast.nan_to_num()

    def _local_search(self, decisions, fit, cv, targets):
        n, d = decisions.shape
        identity = torch.eye(d, dtype=decisions.dtype, device=decisions.device)[None]
        span = self.ub - self.lb
        inverse = identity * span.square()[None, None]
        previous_x = previous_g = None
        for _ in range(self.local_search_steps):
            if self._remaining() < n * (d + 4):
                break
            epsilon = (span * 1e-3).clamp_min(torch.finfo(decisions.dtype).eps)
            signed = torch.where(decisions + epsilon <= self.ub, epsilon, -epsilon)
            probes = clamp(decisions[:, None] + identity * signed[:, None], self.lb, self.ub)
            actual_step = probes.diagonal(dim1=1, dim2=2) - decisions
            probe_fit, probe_cv = self._evaluate(probes.reshape(-1, d))
            objective = (fit - targets).square().sum(1)
            loss = objective if cv is None else torch.where(total_violation(cv) > 0, total_violation(cv), objective)
            probe_objective = (probe_fit.reshape(n, d, -1) - targets[:, None]).square().sum(2)
            if cv is not None:
                violation = total_violation(probe_cv).reshape(n, d)
                probe_objective = torch.where((total_violation(cv) > 0)[:, None], violation, probe_objective)
            gradient = torch.where(
                actual_step != 0, (probe_objective - loss[:, None]) / torch.where(actual_step != 0, actual_step, 1), 0
            )
            if previous_x is not None:
                delta, change = decisions - previous_x, gradient - previous_g
                curvature = (delta * change).sum(1)
                rho = curvature.clamp_min(1e-10).reciprocal()
                transform = identity - rho[:, None, None] * delta[:, :, None] * change[:, None]
                update = (
                    transform @ inverse @ transform.transpose(1, 2) + rho[:, None, None] * delta[:, :, None] * delta[:, None]
                )
                inverse = torch.where((curvature > 1e-10)[:, None, None], update, inverse)
            direction = -(inverse @ gradient[:, :, None]).squeeze(2)
            size = (direction / span.clamp_min(1e-12)).abs().amax(1).clamp_min(1e-12)
            direction = direction / size[:, None]
            alpha = decisions.new_tensor([1.0, 0.5, 0.25, 0.125])
            trials = clamp(decisions[None] + alpha[:, None, None] * direction[None], self.lb, self.ub)
            trial_fit, trial_cv = self._evaluate(trials.reshape(-1, d))
            all_x = torch.cat([decisions, trials.reshape(-1, d)])
            all_fit = torch.cat([fit, trial_fit])
            all_cv = cat_violation(cv, trial_cv)
            distance = (all_fit.reshape(5, n, -1) - targets[None]).square().sum(2)
            keys = [distance] if all_cv is None else [distance, total_violation(all_cv).reshape(5, n)]
            best = lexsort(keys, dim=0)[0]
            selected = best * n + torch.arange(n, device=decisions.device)
            previous_x, previous_g = decisions, gradient
            decisions, fit, cv = all_x[selected], all_fit[selected], take_violation(all_cv, selected)
            if not bool((best != 0).any()):
                break
        return decisions, fit, cv

    def _gmm_vi_fit(self, values, k):
        return _variational_gaussian_mixture(values, k)

    @torch.compiler.disable
    def _predict(self, rank):
        indices, targets = self._forecast_front(rank)
        pop, fit, cv = self._local_search(self.pop[indices], self.fit[indices], take_violation(self.cv, indices), targets)
        count = min(self.pop_size - len(pop), self._remaining())
        if count:
            means, covariance, weights = self._gmm_vi_fit(pop, len(pop))
            extra = clamp(_sample_mixture(means, covariance, weights, count), self.lb, self.ub)
            extra_fit, extra_cv = self._evaluate(extra)
            pop, fit, cv = torch.cat([pop, extra]), torch.cat([fit, extra_fit]), cat_violation(cv, extra_cv)
        if cv is not None or len(pop) < self.pop_size:
            self._select(torch.cat([self.pop, pop]), torch.cat([self.fit, fit]), cat_violation(self.cv, cv))
        else:
            self.pop, self.fit, self.cv = pop, fit, cv

    def step(self) -> None:
        n = self._prepare_step()
        if not n:
            return
        self.t = self.t + 1
        rank = rank_with_constraints(self.fit, self.cv)
        trigger = self._record_history(rank)
        if self._prediction_due(trigger):
            self._predict(rank)
            self.t_init = self.t + 1
            self.p = torch.clamp((self.p.to(self.fit.dtype) / 2).round().long(), min=1)
            return
        donors = torch.randint(0, self.pop_size, (2, n), device=self.pop.device)
        offspring = self.pop[:n] + 0.5 * (self.pop[donors[0]] - self.pop[donors[1]])
        offspring = clamp(polynomial_mutation(offspring, self.lb, self.ub), self.lb, self.ub)
        fit, cv = self._evaluate(offspring)
        self._select(torch.cat([self.pop, offspring]), torch.cat([self.fit, fit]), cat_violation(self.cv, cv))
