import torch
import torch.nn.functional as F
from evox.utils import nanmin


def apd_fn(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    obj: torch.Tensor,
    theta: torch.Tensor,
):
    """
    Compute the APD (Angle-Penalized Distance) based on the given inputs.

    :param x: A tensor representing the indices of the partition.
    :param y: A tensor representing the gamma.
    :param z: A tensor representing the angle.
    :param obj: A tensor of shape (n, m) representing the objectives of the solutions.
    :param theta: A tensor representing the parameter theta used for scaling the reference vector.

    :return: A tensor containing the APD values for each solution.
    """
    selected_z = torch.gather(z, 0, torch.relu(x))
    left = 1 + obj.size(1) * theta * selected_z / y[None, :]
    norm_obj = torch.linalg.vector_norm(obj, dim=1)
    right = norm_obj[x]
    return left * right


def ref_vec_guided(x: torch.Tensor, f: torch.Tensor, v: torch.Tensor, theta: torch.Tensor, cv: torch.Tensor | None = None):
    """
    Perform the Reference Vector Guided Evolutionary Algorithm (RVEA) selection process.

    This function selects solutions based on the Reference Vector Guided Evolutionary Algorithm.
    It calculates the distances and angles between solutions and reference vectors, and returns
    the next set of solutions to be evolved.

    :param x: A tensor of shape (n, d) representing the current population solutions.
    :param f: A tensor of shape (n, m) representing the objective values for each solution.
    :param v: A tensor of shape (r, m) representing the reference vectors.
    :param theta: A tensor representing the parameter theta used in the APD calculation.
    :param cv: Optional constraint violations of shape (n,) or (n, c). Positive values
        are summed. Within each reference-vector partition, feasible solutions compete
        by APD; if none are feasible, minimum total violation wins (PlatEMO RVEA).

    :return: A tuple containing:
        - next_x: The next selected solutions.
        - next_f: The objective values of the next selected solutions.
        - next_cv: Selected violations, only returned when cv is supplied.

    :note:
        The function computes the distances between the solutions and reference vectors,
        and selects the solutions with the minimum APD.
        Empty reference-vector slots are padded with NaNs in every returned tensor.
        Selection uses fixed-shape masks, including for constrained populations.
    """
    nv = v.size(0)

    obj = f - nanmin(f, dim=0, keepdim=True)[0]

    obj = obj.clamp_min(1e-32)

    norm_obj = torch.linalg.vector_norm(obj, dim=1)
    cosine = F.cosine_similarity(v.unsqueeze(1), v.unsqueeze(0), dim=-1)

    cosine = torch.where(
        torch.eye(cosine.size(0), dtype=torch.bool, device=f.device),
        0,
        cosine,
    )
    cosine = cosine.clamp(0.0, 1.0)
    # acos is decreasing: reduce cosines first, then evaluate only nv angles.
    gamma = torch.acos(cosine.max(dim=1).values)
    # Coincident reference vectors must not produce a 0/0 APD.
    gamma = gamma.clamp_min(torch.finfo(f.dtype).eps)

    cosine = F.cosine_similarity(obj.unsqueeze(1), v.unsqueeze(0), dim=-1).clamp(0.0, 1.0)
    angle = torch.acos(cosine)

    nan_mask = torch.isnan(obj).any(dim=1)
    if f.device.type == "cpu":
        # Some Windows/Inductor builds return incorrect indices for this
        # reduction. Separate value/index reductions retain first-index ties.
        selected_angle = angle.amin(dim=1)
        indices = torch.arange(nv, device=f.device)[None, :]
        associate = torch.where(angle == selected_angle[:, None], indices, nv).amin(dim=1).clamp_max(nv - 1)
    else:
        selected_angle, associate = angle.min(dim=1)
    # Compute APD once per solution, rather than gathering an (n, nv) partition
    # matrix and computing distances for every solution/vector pair.
    apd = (1 + f.size(1) * theta * selected_angle / gamma[associate]) * norm_obj
    if cv is not None:
        total_cv = cv.clamp_min(0)
        if cv.ndim == 2:
            total_cv = total_cv.sum(dim=1)
        nan_mask = nan_mask | ~torch.isfinite(total_cv)
    associate = torch.where(nan_mask, -1, associate)
    member = associate[:, None] == torch.arange(nv, device=f.device)[None, :]
    mask_null = ~member.any(dim=0)

    if cv is not None:
        feasible = (total_cv == 0)[:, None]
        has_feasible = (member & feasible).any(dim=0)
        eligible = member & (~has_feasible[None, :] | feasible)
        score = torch.where(has_feasible[None, :], apd[:, None], total_cv[:, None])
    else:
        eligible = member
        score = apd[:, None]
    score = torch.where(eligible, score, torch.inf)
    next_ind = torch.argmin(score, dim=0)
    next_x = torch.where(mask_null.unsqueeze(1), torch.nan, x[next_ind])
    next_f = torch.where(mask_null.unsqueeze(1), torch.nan, f[next_ind])

    if cv is None:
        return next_x, next_f
    cv_mask = mask_null if cv.ndim == 1 else mask_null[:, None]
    next_cv = torch.where(cv_mask, torch.nan, cv[next_ind])
    return next_x, next_f, next_cv
