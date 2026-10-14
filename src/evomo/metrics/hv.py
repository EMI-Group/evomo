import torch


def hv(objs: torch.Tensor, ref: torch.Tensor, num_sample: int = 100000):
    """
    Estimate hypervolume for minimization using Monte Carlo sampling.

    Points outside the reference point contribute no volume. For maximization,
    negate both the objectives and reference point before calling this function.

    :param objs: Objective points of shape (n_points, n_objs).
    :param ref: Reference point of shape (n_objs, ).
    :param num_sample: Number of Monte Carlo samples.
    :return: Estimated hypervolume.
    """

    if num_sample <= 0:
        raise ValueError("num_sample must be positive")
    if objs.size(0) == 0:
        return objs.new_zeros(())

    differences = ref.to(device=objs.device) - objs
    if not differences.is_floating_point():
        differences = differences.to(dtype=torch.get_default_dtype())
    valid = (differences >= 0).all(dim=1, keepdim=True)
    points = torch.where(valid, differences, 0)
    bound = torch.max(points, dim=0).values
    max_vol = torch.prod(bound)
    samples = torch.rand(num_sample, points.size(1), device=points.device, dtype=points.dtype) * bound
    in_hypercube = torch.any(torch.all(samples.unsqueeze(1) < points.unsqueeze(0), dim=2), dim=1)
    hv = in_hypercube.sum() / num_sample * max_vol
    return hv
