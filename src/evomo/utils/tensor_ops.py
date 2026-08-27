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
