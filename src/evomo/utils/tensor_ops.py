"""
evomo.utils.tensor_ops
======================

A collection of small, pure PyTorch tensor utilities used across EvoMO / EvoX-based
multi-objective and evolutionary algorithms.

Design goals:
- Pure PyTorch implementation (no NumPy).
- Deterministic behavior when possible (stable sorting).
- GPU-friendly (avoid Python-side loops over rows).
- Reusable across algorithms without introducing algorithm-specific dependencies.

Typical use cases:
- Unique row extraction
- Lexicographic sorting
- Mask-based indexing utilities
- Tensor-based helper operators for selection / ranking / filtering

Note:
This module is intended to stay lightweight. If some utilities become large or
algorithm-specific, consider moving them to a dedicated submodule.
"""

from __future__ import annotations

from typing import Tuple

import torch
from evox.utils import lexsort


def minmax_normalize(
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    eps: float | None = None,
    extrema: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Normalize columns using the extrema of the selected rows.

    :param x: Floating-point values of shape ``(N, M)``.
    :param mask: Optional boolean mask of shape ``(N,)`` on the input device.
        At least one row must be selected. ``None`` includes every row.
    :param eps: Optional additive offset for the column ranges. ``None`` replaces
        zero ranges by one; a supplied value uses ``range + eps`` instead.
    :param extrema: Optional precomputed ``(minimum, maximum)`` column extrema.
        These are used directly, avoiding extra reductions when callers already
        computed their bounds. Do not combine with ``mask``.
    :returns: A tensor of shape ``(N, M)`` with the input device and dtype.
        With ``eps=None``, constant selected columns use a denominator of one. All rows are
        transformed; excluded rows may lie outside ``[0, 1]``.

    This is column-wise min-max scaling. Algorithms using ideal/nadir history
    or hyperplane intercepts should retain their own normalization rules.
    """
    if extrema is not None:
        if mask is not None:
            raise ValueError("Supply either a mask or precomputed extrema")
        low, high = extrema
    elif mask is None:
        if eps is None:
            low, high = x.amin(0), x.amax(0)
        else:
            # Retain the additive-epsilon callers' original reduction operators
            # so sharing the function does not change Inductor's lowering.
            low = x.min(dim=0, keepdim=True).values
            high = x.max(dim=0, keepdim=True).values
    else:
        low = torch.where(mask[:, None], x, torch.inf).amin(0)
        high = torch.where(mask[:, None], x, -torch.inf).amax(0)
    span = torch.where(high > low, high - low, 1) if eps is None else high - low + eps
    return (x - low) / span


def pairwise_cosine_similarity(
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    *,
    eps: float | None = None,
    mode: str = "normalize",
    clamp: bool = True,
    norms: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute cosine similarities between every pair of row vectors.

    :param x: Floating-point vectors of shape ``(N, D)``.
    :param y: Floating-point vectors of shape ``(K, D)``, with the same device
        and dtype as ``x``. ``None`` compares ``x`` with itself.
    :param eps: Stability constant, defaulting to the input dtype's machine epsilon.
    :param mode: ``"normalize"`` clamps each norm to ``eps`` before normalizing
        the vectors and multiplying. ``"product"`` divides the dot products by
        ``norm_x @ norm_y.T + eps``. ``"offset"`` instead divides by
        ``(norm_x + eps) @ (norm_y + eps).T``. These preserve different algorithms'
        original floating-point evaluation order.
    :param clamp: Whether to clamp the result to ``[-1, 1]``.
    :param norms: Optional precomputed raw row norms of shapes ``(N, 1)`` and
        ``(K, 1)``, allowing algorithms to reuse norms needed elsewhere.
    :returns: Similarities of shape ``(N, K)``. Zero vectors have zero similarity
        with the default positive stability constant.

    Matrix multiplication avoids an ``(N, K, D)`` intermediate.
    """
    if y is None:
        y = x
    if eps is None:
        eps = torch.finfo(x.dtype).eps
    if mode == "normalize":
        if norms is None:
            unit_x = x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
            unit_y = y / y.norm(dim=-1, keepdim=True).clamp_min(eps)
        else:
            unit_x = x / norms[0].clamp_min(eps)
            unit_y = y / norms[1].clamp_min(eps)
        result = unit_x @ unit_y.T
    elif mode == "product":
        if norms is None:
            norm_x = torch.norm(x, dim=1, keepdim=True)
            norm_y = norm_x if y is x else torch.norm(y, dim=1, keepdim=True)
        else:
            norm_x, norm_y = norms
        result = (x @ y.T) / (norm_x @ norm_y.T + eps)
    elif mode == "offset":
        if norms is None:
            norm_x = torch.norm(x, dim=1, keepdim=True) + eps
            norm_y = norm_x if y is x else torch.norm(y, dim=1, keepdim=True) + eps
        else:
            norm_x, norm_y = norms[0] + eps, norms[1] + eps
        result = (x @ y.T) / (norm_x @ norm_y.T)
    else:
        raise ValueError(f"Unknown cosine similarity mode: {mode}")
    return result.clamp(-1, 1) if clamp else result


def pairwise_max_difference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return ``max_d(x[i, d] - y[j, d])`` for all pairs of objective rows."""
    difference = x.unsqueeze(1) - y.unsqueeze(0)
    return torch.max(difference, dim=2)[0]


def shifted_distance(x: torch.Tensor) -> torch.Tensor:
    """Return directed shift-based Euclidean distances for minimization.

    Row ``i`` compares ``x[i]`` with every row shifted up to ``x[i]`` in each
    coordinate. Diagonal entries are zero; callers apply their own masks.
    """
    left = x.unsqueeze(1)
    shifted = torch.maximum(left, x.unsqueeze(0))
    return torch.norm(left - shifted, p=2, dim=2)


def masked_select_indices(mask: torch.Tensor, capacity: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Stably pack true positions into a fixed capacity, returning indices and validity.

    Require a positive mask length and ``0 <= capacity <= len(mask)``. Padding repeats the first selected
    position; if the mask is empty of true entries, padding contains position zero.
    """
    order = torch.argsort((~mask).long(), stable=True)
    slots = torch.arange(capacity, device=mask.device)
    valid = slots < mask.sum()
    return torch.where(valid, order[slots], order[0]), valid


def sample_masked_indices(mask: torch.Tensor, count: int) -> torch.Tensor:
    """Sample selected positions uniformly with replacement; require at least one true entry."""
    order = torch.argsort((~mask).long(), stable=True)
    slots = (torch.rand(count, device=mask.device) * mask.sum().clamp_min(1)).long()
    return order[slots]


def unique_row_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mark the first selected occurrence of each finite objective row, without compaction."""
    order = lexsort(torch.where(mask[:, None], x, torch.inf).T.flip(0).unbind(0))
    first = torch.cat((mask.new_ones(1), (x[order[1:]] != x[order[:-1]]).any(1)))
    return torch.zeros_like(mask).scatter(0, order, first & mask[order])


def at_least_2d(X: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """Ensure the input tensor is at least 2D. Returns (X_2d, was_1d)."""
    if X.ndim == 1:
        return X.unsqueeze(0), True
    elif X.ndim == 2:
        return X, False
    else:
        raise ValueError(f"Expected 1D/2D torch.Tensor, got {tuple(X.shape)}")


def get_pareto_front(f: torch.Tensor) -> torch.Tensor:
    """
    Return the non-dominated set (Pareto front) of the given objectives.
    """
    # f shape: (N, M)
    # Simple O(N^2) implementation
    x_expanded = f.unsqueeze(1)  # (N, 1, M)
    y_expanded = f.unsqueeze(0)  # (1, N, M)

    # x dominates y if x <= y and at least one x < y
    less_equal = (x_expanded <= y_expanded).all(dim=2)
    strictly_less = (x_expanded < y_expanded).any(dim=2)
    domination = less_equal & strictly_less  # (N, N)

    # an element j is dominated if any i dominates it
    is_dominated = domination.any(dim=0)
    return f[~is_dominated]


def unique_rows_sorted(x: torch.Tensor, return_index: bool = True):
    """
    Compute unique rows of a 2D tensor using a **pure sorting-based** approach.

    This function performs a stable lexicographic sort on the rows of `x`
    (from the last column to the first), then removes consecutive duplicates.
    It is a robust replacement for `torch.unique(..., dim=0)` when:
    - certain PyTorch versions do not support required arguments,
    - you want deterministic stable behavior,
    - you want full control over the uniqueness logic.

    Parameters
    ----------
    x : torch.Tensor
        A 2D tensor of shape (N, M), where N is the number of rows and
        M is the number of columns. Each row is treated as a key.
    return_index : bool, default=True
        If True, also return the indices of the unique rows in the **original**
        input tensor `x`.

    Returns
    -------
    uniq : torch.Tensor
        A 2D tensor containing unique rows of `x` after sorting.
        Shape: (K, M), where K <= N.
    uniq_idx : torch.Tensor, optional
        Only returned if `return_index=True`.
        A 1D tensor of shape (K,) containing the original indices of `uniq`
        rows in `x`.

    Notes
    -----
    - Sorting is performed lexicographically using stable sorts column-by-column.
      This guarantees deterministic ordering given deterministic `torch.sort`.
    - Uniqueness is determined by comparing adjacent sorted rows.
    - The output order is the lexicographically sorted order, **not the original order**.

    Complexity
    ----------
    Time: O(M * N log N), where M is number of columns.
    Space: O(N) for indexing buffers.

    Examples
    --------
    >>> x = torch.tensor([[1, 2], [1, 2], [1, 3], [0, 1]])
    >>> uniq, idx = unique_rows_sorted(x, return_index=True)
    >>> uniq
    tensor([[0, 1],
            [1, 2],
            [1, 3]])
    >>> idx
    tensor([3, 0, 2])
    """
    assert x.dim() == 2, f"unique_rows_sorted expects a 2D tensor, got shape={tuple(x.shape)}"

    device = x.device
    N, M = x.shape
    idx = torch.arange(N, device=device)

    # Stable lexicographic sort (last column -> first column)
    for col in range(M - 1, -1, -1):
        key = x[idx, col]
        _, order = torch.sort(key, stable=True)
        idx = idx[order]

    x_sorted = x[idx]

    # Mark changes between adjacent rows
    diff = torch.ones(N, dtype=torch.bool, device=device)
    if N > 1:
        diff[1:] = torch.any(x_sorted[1:] != x_sorted[:-1], dim=1)

    uniq = x_sorted[diff]

    if not return_index:
        return uniq

    uniq_idx = idx[diff]
    return uniq, uniq_idx
