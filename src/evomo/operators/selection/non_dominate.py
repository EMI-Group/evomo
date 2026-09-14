import torch
from evox.utils import lexsort, register_vmap_op


def dominate_relation(x: torch.Tensor, y: torch.Tensor, cv_x: torch.Tensor = None, cv_y: torch.Tensor = None) -> torch.Tensor:
    """Return the domination relation matrix A, where A_{ij} is True if x_i dominates y_j.

    :param x: An array with shape (n1, m) where n1 is the population size and m is the number of objectives.
    :param y: An array with shape (n2, m) where n2 is the population size and m is the number of objectives.

    :returns: The domination relation matrix of x and y.
    """
    # Expand the dimensions of x and y so that we can perform element-wise comparisons
    # Add new dimensions to x and y to prepare them for broadcasting
    x_expanded = x.unsqueeze(1)  # Shape (n1, 1, m)
    y_expanded = y.unsqueeze(0)  # Shape (1, n2, m)

    # Broadcasted comparison: each pair (x_i, y_j)
    less_than_equal = x_expanded <= y_expanded  # Shape (n1, n2, m)
    strictly_less_than = x_expanded < y_expanded  # Shape (n1, n2, m)

    # Check the domination condition: x_i dominates y_j
    domination_matrix = less_than_equal.all(dim=2) & strictly_less_than.any(dim=2)

    # Constraint handling
    if cv_x is not None and cv_y is not None:
        cv_x_expanded = cv_x.unsqueeze(1)
        cv_y_expanded = cv_y.unsqueeze(0)

        cv_x_sum = cv_x_expanded.sum(dim=-1) if cv_x_expanded.ndim > 2 else cv_x_expanded
        cv_y_sum = cv_y_expanded.sum(dim=-1) if cv_y_expanded.ndim > 2 else cv_y_expanded

        cv_x_feasible = (cv_x_sum <= 0)
        cv_y_feasible = (cv_y_sum <= 0)

        case1 = cv_x_feasible & ~cv_y_feasible
        case2 = (~cv_x_feasible) & (~cv_y_feasible) & (cv_x_sum < cv_y_sum)
        case3 = (cv_x_feasible & cv_y_feasible) | ((~cv_x_feasible) & (~cv_y_feasible) & (cv_x_sum == cv_y_sum))

        domination_matrix = case1 | case2 | (case3 & domination_matrix)

    return domination_matrix


def update_dc_and_rank(
    dominate_relation_matrix: torch.Tensor,
    dominate_count: torch.Tensor,
    pareto_front: torch.BoolTensor,
    rank: torch.Tensor,
    current_rank: int,
):
    """
    Update the dominate count and ranks for the current Pareto front.

    :param dominate_relation_matrix: The domination relation matrix between individuals.
    :param dominate_count: The count of how many individuals dominate each individual.
    :param pareto_front: A tensor indicating which individuals are in the current Pareto front.
    :param rank: A tensor storing the rank of each individual.
    :param current_rank: The current Pareto front rank.

    :returns:
        - **rank**: Updated rank tensor.
        - **dominate_count**: Updated dominate count tensor.
    """

    # Update the rank for individuals in the Pareto front
    rank = torch.where(pareto_front, current_rank, rank)
    # Calculate how many individuals in the Pareto front dominate others
    count_desc = torch.sum(pareto_front.unsqueeze(-1) & dominate_relation_matrix, dim=-2, dtype=dominate_count.dtype)

    # Update dominate_count (remove those in the current Pareto front)
    dominate_count = dominate_count - count_desc
    dominate_count = dominate_count - pareto_front.int()

    return rank, dominate_count


def _igr_fake(
    dominate_relation_matrix: torch.Tensor,
    dominate_count: torch.Tensor,
    rank: torch.Tensor,
    pareto_front: torch.Tensor,
    compiling: bool,
) -> torch.Tensor:
    return rank.new_empty(dominate_count.size())


def _igr_fake_vmap(
    dominate_relation_matrix: torch.Tensor,
    dominate_count: torch.Tensor,
    rank: torch.Tensor,
    pareto_front: torch.Tensor,
    compiling: bool,
) -> torch.Tensor:
    return rank.new_empty(dominate_count.size())


def _vmap_iterative_get_ranks_compile(
    dominate_relation_matrix: torch.Tensor,
    dominate_count: torch.Tensor,
    rank: torch.Tensor,
    pareto_front: torch.Tensor,
) -> torch.Tensor:
    def cond_fn(r, cr, dc, pf):
        return pf.any()

    def body_fn(r, cr, dc, pf):
        r, dc = update_dc_and_rank(dominate_relation_matrix, dc, pf, r, cr)
        cr = cr + 1
        new_pareto_front = dc == 0
        pf = torch.where(pf.any(dim=-1, keepdim=True), new_pareto_front, pf)
        return r, cr, dc, pf

    rank = rank.expand_as(dominate_count).contiguous()  # contiguous to unify carry stride
    rank, *_ = torch.while_loop(
        cond_fn, body_fn, (rank, torch.tensor(0, device=rank.device), dominate_count, pareto_front)
    )
    return rank


# evox.core.compile is not necessary since no indexing here
_vmap_iterative_get_ranks_compile = torch.compile(_vmap_iterative_get_ranks_compile, fullgraph=True)


def _vmap_iterative_get_ranks(
    dominate_relation_matrix: torch.Tensor,
    dominate_count: torch.Tensor,
    rank: torch.Tensor,
    pareto_front: torch.Tensor,
    compiling: bool,
) -> torch.Tensor:
    current_rank = 0
    if compiling:
        rank = _vmap_iterative_get_ranks_compile(dominate_relation_matrix, dominate_count, rank, pareto_front)
    else:
        while pareto_front.any():
            rank, dominate_count = update_dc_and_rank(
                dominate_relation_matrix, dominate_count, pareto_front, rank, current_rank
            )
            current_rank += 1
            new_pareto_front = dominate_count == 0
            pareto_front = torch.where(pareto_front.any(dim=-1, keepdim=True), new_pareto_front, pareto_front)
    return rank


def _iterative_get_ranks_compile(
    dominate_relation_matrix: torch.Tensor,
    dominate_count: torch.Tensor,
    rank: torch.Tensor,
    pareto_front: torch.Tensor,
) -> torch.Tensor:
    def cond_fn(r, cr, dc, pf):
        return pf.any()

    def body_fn(r, cr, dc, pf):
        r, dc = update_dc_and_rank(dominate_relation_matrix, dc, pf, r, cr)
        cr = cr + 1
        pf = dc == 0
        return r, cr, dc, pf

    rank, *_ = torch.while_loop(
        cond_fn, body_fn, (rank, torch.tensor(0, device=rank.device), dominate_count, pareto_front)
    )
    return rank


# evox.core.compile is not necessary since no indexing here
_iterative_get_ranks_compile = torch.compile(_iterative_get_ranks_compile, fullgraph=True)


@register_vmap_op(
    fake_fn=_igr_fake, vmap_fn=_vmap_iterative_get_ranks, fake_vmap_fn=_igr_fake_vmap, max_vmap_level=2
)
def _iterative_get_ranks(
    dominate_relation_matrix: torch.Tensor,
    dominate_count: torch.Tensor,
    rank: torch.Tensor,
    pareto_front: torch.Tensor,
    compiling: bool,
) -> torch.Tensor:
    if compiling:
        rank = _iterative_get_ranks_compile(dominate_relation_matrix, dominate_count, rank, pareto_front)
    else:
        current_rank = 0
        while pareto_front.any():
            rank, dominate_count = update_dc_and_rank(
                dominate_relation_matrix, dominate_count, pareto_front, rank, current_rank
            )
            current_rank += 1
            pareto_front = dominate_count == 0
    return rank


def non_dominate_rank(x: torch.Tensor, cv: torch.Tensor = None) -> torch.Tensor:
    """
    Compute the non-domination rank for a set of solutions in multi-objective optimization.

    The non-domination rank is a measure of the Pareto optimality of each solution.

    :param x: A 2D tensor where each row represents a solution, and each column represents an objective.
    :param cv: An optional tensor containing the constraint violations of the solutions.

    :returns:
        A 1D tensor containing the zero-based non-domination rank for each solution.
        Solutions in the first Pareto front have rank 0.
    """

    n = x.size(0)
    # Domination relation matrix (n x n)
    dominate_relation_matrix = dominate_relation(x, x, cv, cv)
    # Count how many times each individual is dominated
    # Counts never exceed n. Narrow integer reductions save CPU memory traffic;
    # retain int64 on accelerators, where int32 did not improve measured latency.
    count_dtype = torch.int32 if x.device.type == "cpu" and n <= torch.iinfo(torch.int32).max else torch.int64
    dominate_count = dominate_relation_matrix.sum(dim=0, dtype=count_dtype)
    # Initialize rank array
    rank = torch.zeros(n, dtype=torch.int32, device=x.device)
    # Identify individuals in the first Pareto front (those that are not dominated)
    pareto_front = dominate_count == 0
    # Iteratively identify Pareto fronts
    rank = _iterative_get_ranks(
        dominate_relation_matrix, dominate_count, rank, pareto_front, torch.compiler.is_compiling()
    )
    return rank


def _partial_rank_fake(
    domination: torch.Tensor,
    count: torch.Tensor,
    rank: torch.Tensor,
    front: torch.Tensor,
    target: torch.Tensor,
    compiling: bool,
) -> torch.Tensor:
    return rank.new_empty(count.shape)


def _partial_rank_compile(domination, count, rank, front, target):
    rank = rank.expand_as(count).contiguous()
    selected = torch.zeros_like(count.sum(dim=-1))
    target = target.expand_as(selected)

    def cond_fn(rank, current_rank, count, front, selected):
        return front.any()

    def body_fn(rank, current_rank, count, front, selected):
        rank, count = update_dc_and_rank(domination, count, front, rank, current_rank)
        selected = selected + front.sum(dim=-1)
        # Finish the entire cutoff front. Each vmap batch stops independently.
        front = (count == 0) & (selected < target).unsqueeze(-1)
        return rank, current_rank + 1, count, front, selected

    rank, *_ = torch.while_loop(
        cond_fn, body_fn,
        (rank, torch.zeros((), dtype=torch.int64, device=rank.device), count, front, selected),
    )
    return rank


_partial_rank_compile = torch.compile(_partial_rank_compile, fullgraph=True)


def _partial_rank_impl(
    domination: torch.Tensor,
    count: torch.Tensor,
    rank: torch.Tensor,
    front: torch.Tensor,
    target: torch.Tensor,
    compiling: bool,
) -> torch.Tensor:
    if compiling:
        return _partial_rank_compile(domination, count, rank, front, target)

    rank = rank.expand_as(count).contiguous()
    selected = torch.zeros_like(count.sum(dim=-1))
    target = target.expand_as(selected)
    current_rank = 0
    while front.any():
        rank, count = update_dc_and_rank(domination, count, front, rank, current_rank)
        selected = selected + front.sum(dim=-1)
        front = (count == 0) & (selected < target).unsqueeze(-1)
        current_rank += 1
    return rank


_partial_rank = register_vmap_op(
    _partial_rank_impl, fake_fn=_partial_rank_fake, vmap_fn=_partial_rank_impl,
    fake_vmap_fn=_partial_rank_fake, max_vmap_level=2,
)


def _environmental_selection_rank(f: torch.Tensor, topk: int, cv: torch.Tensor = None) -> torch.Tensor:
    """Rank through the cutoff front; unvisited ranks use an internal N sentinel."""
    n = f.size(0)
    target = torch.tensor(topk, dtype=torch.int64, device=f.device)
    if cv is not None:
        cv_sum = cv.sum(dim=1) if cv.ndim > 1 else cv
        # Selection sorts by CV before rank. Negative feasible CVs or nonfinite
        # values can violate that ordering across fronts, requiring full ranks.
        full_ranking = ((cv_sum < 0) | ~torch.isfinite(cv_sum)).any()
        target = torch.where(full_ranking, n, target)

    domination = dominate_relation(f, f, cv, cv)
    count_dtype = torch.int32 if f.device.type == "cpu" and n <= torch.iinfo(torch.int32).max else torch.int64
    count = domination.sum(dim=0, dtype=count_dtype)
    rank = torch.full((n,), n, dtype=torch.int32, device=f.device)
    return _partial_rank(domination, count, rank, count == 0, target, torch.compiler.is_compiling())


def crowding_distance(costs: torch.Tensor, mask: torch.Tensor):
    """
    Compute the crowding distance for a set of solutions in multi-objective optimization.

    The crowding distance is a measure of the diversity of solutions within a Pareto front.

    :param costs: A 2D tensor where each row represents a solution, and each column represents an objective.
    :param mask: A 1D boolean tensor indicating which solutions should be considered.

    :returns:
        A 1D tensor containing the crowding distance for each solution.
    """
    total_len = costs.size(0)
    if total_len == 0:
        return costs.new_empty((0,))
    if mask is None:
        mask = torch.ones(total_len, dtype=torch.bool, device=costs.device)
        num_valid_elem = mask.sum()
        # No mask partition is needed when every solution participates.
        if costs.device.type == "cpu":
            order = torch.argsort(costs.T.contiguous(), dim=-1, stable=True).T
        else:
            order = torch.argsort(costs, dim=0, stable=True)
    else:
        num_valid_elem = mask.sum()
        if costs.device.type == "cpu":
            # Sort contiguous objective rows instead of strided columns on CPU.
            inverted_mask = (~mask).unsqueeze(0).expand(costs.size(1), -1)
            order = lexsort([costs.T.contiguous(), inverted_mask], dim=-1).T
        else:
            inverted_mask = (~mask).unsqueeze(1).expand_as(costs).to(costs.dtype)
            order = lexsort([costs, inverted_mask], dim=0)
    sorted_costs = torch.gather(costs, dim=0, index=order)
    last = (num_valid_elem - 1).clamp_min(0).reshape(1, 1).expand(1, costs.size(1))
    span = sorted_costs.gather(0, last) - sorted_costs[:1]
    varying = span > 0
    safe_span = torch.where(varying, span, torch.ones_like(span))
    previous = torch.cat([sorted_costs[:1], sorted_costs[:-1]], dim=0)
    following = torch.cat([sorted_costs[1:], sorted_costs[-1:]], dim=0)
    distance = torch.where(varying, (following - previous) / safe_span, 0)

    # Ignore constant objectives, including their arbitrary endpoints after sorting.
    position = torch.arange(total_len, device=costs.device).unsqueeze(1)
    boundary = (position == 0) | (position == num_valid_elem - 1)
    distance = torch.where(boundary & varying, torch.inf, distance)
    distance = torch.zeros_like(costs).scatter(0, order, distance).sum(dim=1)
    distance = torch.where(num_valid_elem <= 2, torch.inf, distance)
    return torch.where(mask, distance, -torch.inf)


def nd_environmental_selection(x: torch.Tensor, f: torch.Tensor, topk: int, cv: torch.Tensor = None):
    """
    Perform environmental selection based on non-domination rank and crowding distance.

    Ranking stops after the complete front containing the topk-th solution.
    Negative or nonfinite total constraint violations require full ranking to
    preserve the CV-first selection order. Returned ranks are always complete
    for the selected individuals; use non_dominate_rank for all input ranks.

    :param x: A 2D tensor where each row represents a solution, and each column represents a decision variable.
    :param f: A 2D tensor where each row represents a solution, and each column represents an objective.
    :param topk: The number of solutions to select.
    :param cv: An optional tensor containing the constraint violations of the solutions.

    :returns:
        A tuple of five values:
        - **x**: The selected solutions.
        - **f**: The corresponding objective values.
        - **rank**: The non-domination rank of the selected solutions.
        - **crowding_dis**: The crowding distance of the selected solutions.
        - **cv**: The selected constraint violations, or ``None`` for an unconstrained problem.
    """
    # Only the fronts that can survive selection need their actual ranks.
    rank = _environmental_selection_rank(f, topk, cv)
    worst_rank = torch.topk(rank, topk, largest=False)[0][-1]
    mask = rank == worst_rank
    crowding_dis = crowding_distance(f, mask)

    if cv is not None:
        cv_sum = cv.sum(dim=1) if cv.ndim > 1 else cv
        combined_order = lexsort([-crowding_dis, rank, cv_sum])[:topk]
        return x[combined_order], f[combined_order], rank[combined_order], crowding_dis[combined_order], cv[combined_order]
    else:
        combined_order = lexsort([-crowding_dis, rank])[:topk]
        return x[combined_order], f[combined_order], rank[combined_order], crowding_dis[combined_order], None
