import torch
from evox.utils import lexsort


def crowding_distance_by_rank(fit: torch.Tensor, rank: torch.Tensor) -> torch.Tensor:
    """Crowding distance within every front, using batched fixed-shape sorts."""
    size, objectives = fit.shape
    bins = rank[:, None].long().expand(-1, objectives)
    minima = fit.new_full((size, objectives), torch.inf).scatter_reduce(0, bins, fit, reduce="amin")
    maxima = fit.new_full((size, objectives), -torch.inf).scatter_reduce(0, bins, fit, reduce="amax")
    order = lexsort([fit, rank[:, None].expand_as(fit)], dim=0)
    values = fit.gather(0, order)
    fronts = rank[order]
    span = maxima.gather(0, fronts.long()) - minima.gather(0, fronts.long())
    varying = span > 0
    previous = torch.cat([values[:1], values[:-1]])
    following = torch.cat([values[1:], values[-1:]])
    left_edge = torch.cat([torch.ones_like(fronts[:1], dtype=torch.bool), fronts[1:] != fronts[:-1]])
    right_edge = torch.cat([fronts[:-1] != fronts[1:], torch.ones_like(fronts[:1], dtype=torch.bool)])
    distance = torch.where(varying, (following - previous) / torch.where(varying, span, 1), 0)
    distance = torch.where((left_edge | right_edge) & varying, torch.inf, distance)
    distance = torch.zeros_like(fit).scatter(0, order, distance).sum(dim=1)
    counts = torch.zeros(size, device=fit.device, dtype=torch.long).scatter_add(0, rank.long(), torch.ones_like(rank).long())
    return torch.where(counts[rank.long()] <= 2, torch.inf, distance)


def distance_truncation(distance: torch.Tensor, count: int, neighbors: int | None = None) -> torch.Tensor:
    """Keep ``count`` rows using PlatEMO's iterative distance truncation.

    Recompute distances to active rows after every deletion. With ``neighbors``
    use the product of the nearest distances; otherwise compare all sorted
    distances lexicographically. Masks retain the original matrix shape.
    """
    size = distance.shape[0]
    if not 0 <= count <= size:
        raise ValueError("count must be between zero and the number of rows")
    diagonal = torch.eye(size, dtype=torch.bool, device=distance.device)
    distance = distance.masked_fill(diagonal, torch.inf)
    active = torch.ones(size, dtype=torch.bool, device=distance.device)
    for _ in range(size - count):
        current = distance.masked_fill(~active[None], torch.inf)
        if neighbors is None:
            ordered = current.sort(dim=1).values
            if distance.device.type == "cuda" and size <= 128:
                # A bounded comparison avoids launching one tiny sort per
                # distance column for small GPU populations. Both paths use
                # the same exact lexicographic order, including index ties.
                different = ordered[:, None] != ordered[None]
                first = different.to(torch.int32).argmax(dim=-1)
                left = ordered[:, None].expand(-1, size, -1).gather(2, first[:, :, None]).squeeze(2)
                right = ordered[None].expand(size, -1, -1).gather(2, first[:, :, None]).squeeze(2)
                unequal = different.any(dim=-1)
                indices = torch.arange(size, device=distance.device)
                earlier = (unequal & (left < right)) | (~unequal & (indices[:, None] < indices[None]))
                preceded = (earlier & active[:, None]).any(dim=0)
                remove = torch.where(active & ~preceded, indices, size).amin()
            else:
                keys = torch.cat([ordered.T.flip(0), (~active)[None].to(distance.dtype)])
                remove = lexsort(keys)[0]
        else:
            closest = current.topk(min(neighbors, size), largest=False, dim=1).values
            score = torch.nan_to_num(closest.prod(dim=1), nan=torch.inf, posinf=torch.inf)
            score = torch.where(active, score, torch.inf)
            remove = torch.where(torch.isinf(score.amin()), active.to(torch.int32).argmax(), score.argmin())
        active = active.scatter(0, remove.reshape(1), False)
    return active
