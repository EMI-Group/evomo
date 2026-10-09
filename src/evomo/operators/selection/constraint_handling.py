"""Fixed-shape constraint comparisons shared by constrained algorithm extensions."""

import torch
from evox.utils import lexsort

from .non_dominate import _environmental_selection_rank, crowding_distance, non_dominate_rank


def total_violation(cv):
    """Sum positive violations, accepting scalar totals or per-constraint columns."""
    positive = torch.nan_to_num(cv.clamp_min(0), nan=torch.inf, posinf=torch.inf, neginf=0)
    return positive.sum(dim=1) if cv.ndim == 2 else positive


def cat_violation(*values):
    """Concatenate optional constraint state without allocating on the none path."""
    return None if values[0] is None else torch.cat(values, dim=0)


def take_violation(cv, indices):
    return None if cv is None else cv[indices]


def rank_with_constraints(fit, cv=None):
    return non_dominate_rank(fit) if cv is None else constrained_rank(fit, cv)


def rank_with_constraint_objective(fit, cv=None):
    """Joint objective/CV fronts for exploratory probes and auxiliary archives."""
    if cv is None:
        return non_dominate_rank(fit)
    return non_dominate_rank(torch.cat([fit, total_violation(cv)[:, None]], dim=1))


def constraint_keys(keys, cv):
    """Append CV as the primary lexicographic key; keep the original none path."""
    return keys if cv is None else [*keys, total_violation(cv)]


def constraint_dominance_matrix(dominance, cv, other_cv=None, *, objective_ties=False):
    """Apply constraint dominance to an existing row-dominates-column matrix.

    Some strength-based PlatEMO selectors compare objectives at equal positive
    violation too; request that explicitly using ``objective_ties=True``.
    """
    if cv is None:
        return dominance
    left = total_violation(cv)[:, None]
    right = total_violation(cv if other_cv is None else other_cv)[None, :]
    ties = left == right if objective_ties else (left == 0) & (right == 0)
    return (left < right) | (ties & dominance)


def worst_constraint_index(scores, active, cv=None):
    """Delete highest CV first, using minimum native survival score at ties."""
    eligible = active
    if cv is not None:
        total = total_violation(cv)
        eligible = active & (total == torch.where(active, total, -torch.inf).amax())
    return torch.where(eligible, scores, torch.inf).argmin()


def constrained_crowding_selection(fit, cv, count):
    """Return indices and rank/distance state for a constrained front cutoff."""
    rank = constrained_rank(fit, cv, count)
    cutoff = rank.kthvalue(count).values
    distance = crowding_distance(fit, rank == cutoff)
    indices = lexsort([-distance, rank])[:count]
    return indices, rank[indices], distance[indices]


def prefer_by_constraint(objective_better, candidate_total, incumbent_total):
    """C-MOEA/D comparison; broadcasting follows the scalarization matrix."""
    return (candidate_total < incumbent_total) | ((candidate_total == incumbent_total) & objective_better)


def constraint_improvement(objective_delta, old_total, new_total):
    """Credit violation changes first and native objective progress at equal CV."""
    relative = (old_total - new_total) / torch.where(old_total > 0, old_total, 1)
    relative = torch.nan_to_num(relative, nan=1, posinf=1, neginf=-1)
    return torch.where(old_total != new_total, relative, objective_delta)


def constrained_dominates(fit, cv, other_fit, other_cv):
    """Deb's constraint dominance, with objective dominance only when feasible."""
    total, other_total = total_violation(cv), total_violation(other_cv)
    pareto = (fit <= other_fit).all(dim=-1) & (fit < other_fit).any(dim=-1)
    return (total < other_total) | ((total == 0) & (other_total == 0) & pareto)


def update_by_proposals(pop, fit, cv, off_pop, off_fit, off_cv, targets, better, scores, limit=None):
    """Resolve fixed-shape neighborhood proposals by (CV, scalar value, source).

    Multiple offspring can propose the same incumbent. Explicit reductions avoid
    conflicting CUDA writes and keep decision/objective/constraint rows aligned.
    ``targets`` and the comparison/score matrices have shape (offspring, slots).
    """
    if limit is not None:
        better = better & (better.long().cumsum(dim=1) <= limit)
    n = pop.shape[0]
    totals = total_violation(off_cv)
    flat_targets = targets.reshape(-1).long()
    proposal_cv = torch.where(better, totals[:, None], torch.inf).reshape(-1)
    best_cv = totals.new_full((n,), torch.inf).scatter_reduce(0, flat_targets, proposal_cv, reduce="amin")
    eligible = better & (totals[:, None] == best_cv[targets])
    proposal_score = torch.where(eligible, scores, torch.inf).reshape(-1)
    best_score = scores.new_full((n,), torch.inf).scatter_reduce(0, flat_targets, proposal_score, reduce="amin")
    eligible = eligible & (scores == best_score[targets])
    source = torch.arange(off_pop.shape[0], device=pop.device)[:, None].expand_as(targets)
    sentinel = off_pop.shape[0]
    proposal_source = torch.where(eligible, source, sentinel).reshape(-1)
    chosen = source.new_full((n,), sentinel).scatter_reduce(0, flat_targets, proposal_source, reduce="amin")
    replace = chosen < sentinel
    chosen = chosen.clamp_max(sentinel - 1)
    new_cv = torch.where(replace.reshape((-1,) + (1,) * (cv.ndim - 1)), off_cv[chosen], cv)
    return torch.where(replace[:, None], off_pop[chosen], pop), torch.where(replace[:, None], off_fit[chosen], fit), new_cv


def update_last_proposals(pop, fit, off_pop, off_fit, targets, active):
    """Resolve objective-only batch collisions with the intended last-source rule."""
    source = torch.arange(off_pop.shape[0], device=pop.device)[:, None].expand_as(targets)
    proposals = torch.where(active, source, -1).reshape(-1)
    chosen = source.new_full((pop.shape[0],), -1).scatter_reduce(0, targets.reshape(-1).long(), proposals, reduce="amax")
    replace = chosen >= 0
    chosen = chosen.clamp_min(0)
    return torch.where(replace[:, None], off_pop[chosen], pop), torch.where(replace[:, None], off_fit[chosen], fit)


def constraint_priority(fitness, cv):
    """Ascending tournament keys: violation first, then fitness, preserving ties."""
    total = total_violation(cv)
    order = lexsort([fitness, total])
    values, violations = fitness[order], total[order]
    distinct = torch.cat(
        (
            torch.ones(1, dtype=torch.bool, device=fitness.device),
            (values[1:] != values[:-1]) | (violations[1:] != violations[:-1]),
        )
    )
    priority = distinct.long().cumsum(0) - 1
    return torch.zeros_like(priority).scatter(0, order, priority)


def constrained_rank(fit, cv, count=None):
    """PlatEMO NDSort semantics: equal-CV infeasible points share a front.

    Sort feasible objectives with fixed-shape padding, then append dense CV ranks.
    Sorting violations avoids peeling one front per distinct CV in all-infeasible
    populations. Equal-CV points remain tied regardless of objective dominance.
    """
    total = total_violation(cv)
    feasible = total == 0
    rank_fit = torch.where(feasible[:, None], fit, torch.inf)
    if count is None:
        feasible_rank = non_dominate_rank(rank_fit)
    else:
        feasible_rank = _environmental_selection_rank(rank_fit, count)
    n = fit.shape[0]
    offset = torch.where(feasible & (feasible_rank < n), feasible_rank, -1).amax() + 1
    order = total.argsort(stable=True)
    sorted_cv = total[order]
    distinct = torch.cat((torch.ones(1, dtype=torch.bool, device=fit.device), sorted_cv[1:] != sorted_cv[:-1]))
    dense = (distinct & (sorted_cv > 0)).cumsum(0, dtype=feasible_rank.dtype) - 1
    violation_rank = torch.zeros_like(feasible_rank).scatter(0, order, dense)
    rank = torch.where(feasible, feasible_rank, offset + violation_rank)
    if count is not None:
        cutoff = rank.kthvalue(count).values
        rank = torch.where(rank <= cutoff, rank, n)
    return rank
