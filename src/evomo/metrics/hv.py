from numbers import Integral

import torch


def hv(objs: torch.Tensor, ref: torch.Tensor, num_sample: int = 100000):
    """
    Estimate hypervolume for minimization using Monte Carlo sampling.

    Nonfinite objective rows and points outside or on the reference boundary
    contribute no volume. A nonfinite reference point returns NaN. For
    maximization, negate both the objectives and reference point.

    Float16, bfloat16 and integral inputs are promoted to at least float32.
    All samples are evaluated together using broadcast tensor operations.
    This is a stochastic estimate in the input objective scales; no normalization
    is applied automatically.

    :param objs: Objective points of shape (n_points, n_objs).
    :param ref: Reference point of shape (n_objs, ).
    :param num_sample: Number of Monte Carlo samples.
    :return: Estimated hypervolume.
    """

    if isinstance(num_sample, bool) or not isinstance(num_sample, Integral):
        raise TypeError("num_sample must be an integer")
    if num_sample <= 0:
        raise ValueError("num_sample must be positive")
    if objs.ndim != 2 or objs.shape[1] == 0:
        raise ValueError("objs must have shape (n_points, n_objs), with n_objs > 0")
    if ref.ndim != 1 or ref.shape[0] != objs.shape[1]:
        raise ValueError("ref must have shape (n_objs,) matching objs")
    if objs.is_complex() or ref.is_complex():
        raise TypeError("objectives and reference point must be real-valued")

    dtype = torch.promote_types(objs.dtype, ref.dtype)
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    reference = ref.to(device=objs.device, dtype=dtype)
    reference_is_finite = reference.isfinite().all()
    invalid = reference.new_full((), torch.nan)
    if objs.shape[0] == 0:
        return torch.where(reference_is_finite, reference.new_zeros(()), invalid)

    objectives = objs.to(dtype=dtype)
    differences = reference - objectives
    valid = (objectives.isfinite() & (differences > 0)).all(dim=1, keepdim=True)
    points = torch.where(valid, differences, 0)
    bound = torch.max(points, dim=0).values
    max_vol = torch.prod(bound)
    samples = torch.rand(num_sample, points.size(1), device=points.device, dtype=dtype) * bound
    in_hypercube = torch.any(torch.all(samples.unsqueeze(1) < points.unsqueeze(0), dim=2), dim=1)
    volume = in_hypercube.mean(dtype=dtype) * max_vol
    return torch.where(reference_is_finite, volume, invalid)
