import torch
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary_half
from evox.operators.selection import tournament_selection_multifit
from evox.utils import clamp

from evomo.operators.selection.constraint_handling import (
    cat_violation,
    take_violation,
)
from evomo.utils import register_lazy_buffer

from .sparseea import (
    _evaluate_sparse,
    _initialize_sparse,
    _select_sparse,
    _sparse_mask_variation,
    _sparse_offspring_count,
)
from .sparseea2 import _grouped_real_mutation


def _train_mlp(history, active, hidden=10, epochs=500):
    """Batched implementation of DM-MOEA's explicitly trained sigmoid network."""
    time, count, dim = history.shape
    result = history.new_zeros((count, dim))
    if time < 2 or not bool(active.any()):
        return history[-1].clone()
    data = history[:, :, active].transpose(0, 1)
    x, target = data[:, :-1], data[:, -1:]
    features = data.shape[-1]
    w1 = torch.randn(count, features, hidden, dtype=history.dtype, device=history.device)
    w2 = torch.randn(count, hidden, features, dtype=history.dtype, device=history.device)
    b1 = torch.randn(count, 1, hidden, dtype=history.dtype, device=history.device)
    b2 = torch.randn(count, 1, features, dtype=history.dtype, device=history.device)
    for _ in range(epochs):
        latent = torch.sigmoid(x @ w1 + b1)
        output = torch.sigmoid(latent @ w2 + b2)
        delta = output - target
        hidden_delta = (delta @ w2.transpose(1, 2)) * latent * (1 - latent)
        dw2, dw1 = latent.transpose(1, 2) @ delta, x.transpose(1, 2) @ hidden_delta
        w2, w1 = w2 - 0.1 * dw2, w1 - 0.1 * dw1
        b2, b1 = b2 - 0.1 * delta.sum(1, keepdim=True), b1 - 0.1 * hidden_delta.sum(1, keepdim=True)
    result[:, active] = torch.sigmoid(torch.sigmoid(data[:, -1:] @ w1 + b1) @ w2 + b2).squeeze(1)
    return result


def _predict_svr_mask(history, iterations=200, epsilon=0.1, box=1.0):
    """RBF epsilon-SVR dual with the equality and box constraints enforced by proximal projection."""
    data = history.transpose(0, 1)
    x, y = data[:, :-1], data[:, 1:]
    n, time, dim = x.shape
    varying = data.amin(1) != data.amax(1)
    x = x * varying[:, None]
    query = data[:, -1:] * varying[:, None]
    kernel = torch.exp(-0.5 * torch.cdist(x, x).square())
    query_kernel = torch.exp(-0.5 * torch.cdist(query, x).square())
    coefficient = torch.zeros_like(y)
    step = 1 / kernel.sum(2).amax(1).clamp_min(1)
    for _ in range(iterations):
        candidate = coefficient - step[:, None, None] * (kernel @ coefficient - y)
        lower, upper = candidate.amin(1, keepdim=True) - box, candidate.amax(1, keepdim=True) + box
        for _ in range(16):
            middle = (lower + upper) / 2
            centered = candidate - middle
            projected = centered.sign() * (centered.abs() - step[:, None, None] * epsilon).clamp_min(0)
            projected = projected.clamp(-box, box)
            positive = projected.sum(1, keepdim=True) > 0
            lower, upper = torch.where(positive, middle, lower), torch.where(positive, upper, middle)
        coefficient = projected
    residual = y - kernel @ coefficient - epsilon * coefficient.sign()
    free = (coefficient.abs() > 1e-4) & (coefficient.abs() < box - 1e-4)
    bias = torch.where(free, residual, 0).sum(1) / free.sum(1).clamp_min(1)
    bias = torch.where(free.any(1), bias, residual.mean(1))
    prediction = (query_kernel @ coefficient).squeeze(1) + bias
    sampled = prediction > torch.rand(n, dim, dtype=data.dtype, device=data.device)
    return torch.where(varying, sampled, data[:, -1].bool())


class DMMOEA(Algorithm):
    def __init__(self, pop_size: int, n_objs: int, lb: torch.Tensor, ub: torch.Tensor, **kwargs):
        """Initialize the DMMOEA population and optimization state.

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
        :param kwargs: Optional ``max_fe`` bounds actual evaluations, including probes and initialization.
            Place bounds on the intended device; ``device=...`` does not move them.
            ``detect_change=False`` disables dynamic-environment detection for static-only use.
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
        self.P = 4  # Number of preceding environment snapshots
        self.H = 10  # Hidden size for MLP

        # Initialize State (Mutables)
        self.pop = Mutable(torch.rand(pop_size, self.dim, device=device) * (ub - lb) + lb)
        self.mask = Mutable(torch.rand(pop_size, self.dim, device=device) < 0.5)
        self.fit = Mutable(torch.full((pop_size, n_objs), torch.inf, device=device))
        self.var_fit = Mutable(torch.zeros(self.dim, device=device))

        # History Buffers
        self.dec_history = Mutable(torch.zeros(self.P + 1, pop_size, self.dim, dtype=lb.dtype, device=device))
        self.mask_history = Mutable(torch.zeros(self.P + 1, pop_size, self.dim, dtype=lb.dtype, device=device))
        self.iter_count = Mutable(torch.tensor(0, dtype=torch.int32, device=device))
        self.dis = Mutable(torch.zeros(pop_size, device=device))
        self.rank = Mutable(torch.full((pop_size,), torch.iinfo(torch.int32).max, dtype=torch.int32, device=device))

        register_lazy_buffer(self, "cv", device_like="pop")
        self.fe = Mutable(torch.tensor(0, dtype=torch.int64, device=device))
        self.change_count = Mutable(torch.tensor(0, dtype=torch.int64, device=device))
        self.history_count = Mutable(torch.tensor(1, dtype=torch.int64, device=device))
        self.max_fe = kwargs.get("max_fe")
        self.detect_change = kwargs.get("detect_change", True)

    def init_step(self) -> None:
        _initialize_sparse(self)
        self.dec_history[0] = self.pop
        self.mask_history[0] = self.mask.to(self.pop.dtype)

    def _remaining(self):
        return 2**31 if self.max_fe is None else max(0, self.max_fe - int(self.fe))

    def _batched_mlp_predict(self, history):
        return _train_mlp(history, self.mask.any(0), hidden=self.H)

    def _batched_krr_predict(self, history):
        # Kept as a private compatibility name; the model is now epsilon-SVR, not kernel ridge regression.
        return _predict_svr_mask(history)

    def _environment_changed(self):
        count = min((self.pop_size + 9) // 10, self._remaining())
        if not count:
            return False
        indices = torch.randperm(len(self.pop), device=self.pop.device)[:count]
        fit, cv = _evaluate_sparse(self, self.pop[indices] * self.mask[indices])
        self._last_change_probe = indices, fit, cv
        changed = not torch.equal(fit, self.fit[indices])
        if (cv is None) != (self.cv is None):
            return True
        return changed or (cv is not None and not torch.equal(cv, self.cv[indices]))

    def _respond_to_change(self):
        if self._remaining() < self.pop_size:
            indices, fit, cv = self._last_change_probe
            repeat = torch.arange(self.pop_size, device=self.pop.device) % len(indices)
            self.pop, self.mask, self.fit, self.cv, self.rank, self.dis = _select_sparse(
                self.pop[indices][repeat], self.mask[indices][repeat], fit[repeat], take_violation(cv, repeat), self.pop_size
            )
            self.change_count = self.change_count + 1
            return
        # All incumbent caches must describe the new environment before selection.
        self.fit, self.cv = _evaluate_sparse(self, self.pop * self.mask)
        self.pop, self.mask, self.fit, self.cv, self.rank, self.dis = _select_sparse(
            self.pop, self.mask, self.fit, self.cv, self.pop_size
        )
        self.change_count = self.change_count + 1
        count = int(self.history_count)
        if count == self.P + 1:
            self.dec_history = torch.cat([self.dec_history[1:], self.pop[None]])
            self.mask_history = torch.cat([self.mask_history[1:], self.mask.to(self.pop.dtype)[None]])
        else:
            self.dec_history[count] = self.pop
            self.mask_history[count] = self.mask.to(self.pop.dtype)
            self.history_count = self.history_count + 1
        count = int(self.history_count)
        history = self.dec_history[:count]
        mask = self.mask if int(self.change_count) < self.P else self._batched_krr_predict(self.mask_history[:count])
        decisions = clamp(_train_mlp(history, mask.any(0), hidden=self.H), self.lb, self.ub)
        size = min(self.pop_size, self._remaining())
        if not size:
            return
        decisions, mask = decisions[:size], mask[:size]
        fit, cv = _evaluate_sparse(self, decisions * mask)
        self.pop, self.mask, self.fit, self.cv, self.rank, self.dis = _select_sparse(
            torch.cat([self.pop, decisions]),
            torch.cat([self.mask, mask]),
            torch.cat([self.fit, fit]),
            cat_violation(self.cv, cv),
            self.pop_size,
        )

    def step(self) -> None:
        if self._remaining() == 0:
            return
        self.iter_count = self.iter_count + 1
        if self.detect_change and self._environment_changed():
            self._respond_to_change()
        n = _sparse_offspring_count(self)
        if not n:
            return
        mating = tournament_selection_multifit(2 * n, [-self.dis, self.rank.float()], tournament_size=2)
        masks = self.mask[mating]
        decisions = simulated_binary_half(self.pop[mating], pro_c=1.0, dis_c=20.0)
        decisions, groups = _grouped_real_mutation(decisions, self.lb, self.ub)
        mask = _sparse_mask_variation(masks[:n], masks[n:], self.var_fit, groups)
        fit, cv = _evaluate_sparse(self, decisions * mask)
        self.pop, self.mask, self.fit, self.cv, self.rank, self.dis = _select_sparse(
            torch.cat([self.pop, decisions]),
            torch.cat([self.mask, mask]),
            torch.cat([self.fit, fit]),
            cat_violation(self.cv, cv),
            self.pop_size,
        )
