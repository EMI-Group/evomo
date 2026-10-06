"""Objective-space metrics for minimization at both levels of a bilevel problem."""

import math

import torch


def _inputs(objs, reference, mask):
    if objs.ndim < 2 or objs.shape[-1] == 0 or not objs.is_floating_point():
        raise ValueError("objs must be floating point with shape (..., n, m), m > 0.")
    if reference.device != objs.device or reference.dtype != objs.dtype:
        raise ValueError("The reference must have the same device and dtype as objs.")
    if reference.shape[-1:] != objs.shape[-1:]:
        raise ValueError("The reference and objs must have the same objective count.")
    if mask is None:
        return torch.ones_like(objs[..., 0], dtype=torch.bool)
    if mask.shape != objs.shape[:-1] or mask.dtype != torch.bool or mask.device != objs.device:
        raise ValueError("mask must be boolean on the same device, with shape objs.shape[:-1].")
    return mask


def lgd(objs: torch.Tensor, pf: torch.Tensor, *, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Lower-level generational distance: ``sqrt(sum_i d_i**2) / n``.

    ``objs`` has shape ``(..., n, m_l)``. For joint leader/follower pairs, use
    a conditional reference ``pf`` of shape ``(..., n, r, m_l)``: point i is
    compared only with the lower PF for its own leader. For one fixed leader,
    a shared ``(r, m_l)`` front is also accepted. These are objective values,
    not decision variables. Distances use the same convention as ``gd``.

    ``mask`` selects points without changing tensor shapes. Each batch returns
    one score, or infinity if no point is selected. An empty reference raises
    ValueError. Selected nonfinite points give infinity. Reference values must
    be finite. No normalization or feasibility/optimality check is performed.
    """
    selected = _inputs(objs, pf, mask)
    paired = pf.ndim == objs.ndim + 1 and pf.shape[:-2] == objs.shape[:-1]
    if not paired and pf.ndim != 2:
        raise ValueError("pf must be (r, m_l) or (..., n, r, m_l), matched to objs.")
    if pf.shape[-2] == 0:
        raise ValueError("pf must contain at least one reference point.")
    if objs.shape[-2] == 0:
        return objs.new_full(objs.shape[:-2], torch.inf)
    finite = torch.isfinite(objs).all(-1)
    safe = torch.where(finite.unsqueeze(-1), objs, 0)
    if paired:
        distances = torch.cdist(safe.unsqueeze(-2), pf, compute_mode="donot_use_mm_for_euclid_dist").squeeze(-2)
    else:
        distances = torch.cdist(safe, pf, compute_mode="donot_use_mm_for_euclid_dist")
    nearest = torch.where(finite, distances.amin(-1), torch.inf)
    nearest = torch.where(selected, nearest, 0)
    count = selected.sum(-1)
    score = torch.linalg.vector_norm(nearest, dim=-1) / count.clamp_min(1).to(objs.dtype)
    return torch.where(count > 0, score, torch.inf)


def uigd(objs: torch.Tensor, pf: torch.Tensor, p: float = 1, *, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Upper-level IGD: ``mean_r(min_i ||pf_r - objs_i||_2**p)**(1/p)``.

    ``objs`` has shape ``(..., n, m_u)``; ``pf`` is a finite shared ``(r, m_u)``
    reference or a batch-matched ``(..., r, m_u)`` reference. The default p=1
    agrees with ``igd``. Lower scores are better. No implicit normalization,
    nondominated sorting, or lower-level optimality check is performed.

    Use ``mask`` to exclude infeasible pairs. Nonfinite approximation points
    are ignored; no remaining point gives infinity. An empty PF raises
    ValueError. The result preserves the batch shape, device and dtype.
    """
    selected = _inputs(objs, pf, mask)
    if not math.isfinite(p) or p <= 0:
        raise ValueError("p must be finite and positive.")
    if pf.ndim != 2 and not (pf.ndim == objs.ndim and pf.shape[:-2] == objs.shape[:-2]):
        raise ValueError("pf must be (r, m_u) or (..., r, m_u), matched to objs.")
    if pf.ndim < 2 or pf.shape[-2] == 0:
        raise ValueError("pf must contain at least one reference point.")
    if objs.shape[-2] == 0:
        return objs.new_full(objs.shape[:-2], torch.inf)
    selected = selected & torch.isfinite(objs).all(-1)
    safe = torch.where(selected.unsqueeze(-1), objs, 0)
    distances = torch.cdist(pf, safe, compute_mode="donot_use_mm_for_euclid_dist")
    nearest = torch.where(selected.unsqueeze(-2), distances, torch.inf).amin(-1)
    return nearest.pow(p).mean(-1).pow(1 / p)


def uhv(
    objs: torch.Tensor,
    ref: torch.Tensor,
    num_sample: int = 100000,
    *,
    mask: torch.Tensor | None = None,
    chunk_size: int = 4096,
    point_chunk_size: int = 256,
    samples: torch.Tensor | None = None,
) -> torch.Tensor:
    """Estimate upper hypervolume by uniform Monte Carlo sampling.

    ``objs`` has shape ``(..., n, m_u)`` and the finite reference point ``ref``
    is ``(m_u,)`` or batch-matched ``(..., m_u)``. The sampling box runs from
    the coordinate-wise minimum of eligible points to ref. The result is box
    volume times the fraction dominated by at least one eligible point.

    ``mask`` excludes infeasible pairs. Nonfinite points and points outside ref
    contribute zero; an empty set gives zero. Duplicates and dominated points
    need no preprocessing. Larger scores are better. No normalization or
    lower-level optimality check is performed.

    Both samples and points are chunked to bound temporary dominance storage.
    By default samples are drawn using the PyTorch RNG on objs.device. For
    reproducible comparisons, provide unit-box ``samples`` in [0, 1), of shape
    ``(num_sample, m_u)`` or ``(..., num_sample, m_u)``. Values are rescaled to
    the sampling box; device and dtype must match objs. Fixed shapes, masks
    and chunk sizes support torch.compile; vmap needs an explicit randomness
    policy when samples are generated internally.
    """
    selected = _inputs(objs, ref, mask)
    for name, value in (("num_sample", num_sample), ("chunk_size", chunk_size), ("point_chunk_size", point_chunk_size)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if ref.ndim != 1 and ref.shape != (*objs.shape[:-2], objs.shape[-1]):
        raise ValueError("ref must be (m_u,) or (..., m_u), matched to objs.")
    if samples is not None:
        if samples.device != objs.device or samples.dtype != objs.dtype:
            raise ValueError("samples must have the same device and dtype as objs.")
        shared_shape = (num_sample, objs.shape[-1])
        if samples.shape != shared_shape and samples.shape != (*objs.shape[:-2], *shared_shape):
            raise ValueError("samples must be (num_sample, m_u) or batch-matched (..., num_sample, m_u).")
    if objs.shape[-2] == 0:
        return objs.new_zeros(objs.shape[:-2])
    differences = ref.unsqueeze(-2) - objs
    selected = selected & torch.isfinite(objs).all(-1) & (differences >= 0).all(-1)
    points = torch.where(selected.unsqueeze(-1), differences, 0)
    bound = points.amax(-2)
    volume = bound.prod(-1)
    covered_count = torch.zeros_like(volume, dtype=torch.int64)
    for start in range(0, num_sample, chunk_size):
        size = min(chunk_size, num_sample - start)
        if samples is None:
            unit = torch.rand((*objs.shape[:-2], size, objs.shape[-1]), device=objs.device, dtype=objs.dtype)
        else:
            unit = samples[..., start : start + size, :]
        positions = unit * bound.unsqueeze(-2)
        covered = torch.zeros_like(positions[..., 0], dtype=torch.bool)
        for point_start in range(0, objs.shape[-2], point_chunk_size):
            boxes = points[..., point_start : point_start + point_chunk_size, :]
            dominated = (positions.unsqueeze(-2) < boxes.unsqueeze(-3)).all(-1).any(-1)
            covered = covered | dominated
        covered_count = covered_count + covered.sum(-1)
    return volume * (covered_count.to(objs.dtype) / num_sample)
