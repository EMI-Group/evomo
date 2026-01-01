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

import torch


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
