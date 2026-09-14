"""Experimental single-population Triton non-dominated sorting.

CUDA float32/float64 ranking uses packed dominance and integer rank updates.
CPU inputs use the PyTorch backend. Crowding and final stable sorting remain
in PyTorch. No vmap rules are provided. Fullgraph compilation is supported,
but recompilation and small end-to-end floating-point differences can occur.
"""

import torch
import triton
import triton.language as tl
from evox.utils import lexsort

from evomo.operators.selection import non_dominate as reference

if not all(hasattr(torch.library, name) for name in ("custom_op", "triton_op", "wrap_triton")):
    raise ImportError("The Triton backend requires PyTorch with torch.library.triton_op and wrap_triton support.")


def _check_input(f, cv=None):
    if f.ndim != 2:
        raise ValueError("The Triton backend accepts a single population with shape (N, M).")
    if f.device.type == "cuda" and f.dtype not in (torch.float32, torch.float64):
        raise TypeError("The Triton backend supports CUDA float32 and float64 objectives.")
    if cv is not None:
        if cv.ndim not in (1, 2) or cv.shape[0] != f.shape[0]:
            raise ValueError("Constraint violations must have shape (N,) or (N, C).")
        if cv.device != f.device:
            raise ValueError("Objectives and constraint violations must be on the same device.")


@triton.jit
def _popcount(x):
    return tl.inline_asm_elementwise("popc.b32 $0, $1;", constraints="=r,r", args=[x],
                                   dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _pack_dom(F, CV, D, N: tl.constexpr, M: tl.constexpr, W: tl.constexpr,
              HAS_CV: tl.constexpr, BS: tl.constexpr, BT: tl.constexpr):
    # D[target, source_word]: source bits dominating each target.
    s = tl.program_id(0) * BS + tl.arange(0, BS)
    t = tl.program_id(1) * BT + tl.arange(0, BT)
    b = tl.program_id(2)
    le = tl.full((BT, BS), True, tl.int1)
    lt = tl.full((BT, BS), False, tl.int1)
    for j in range(M):
        a = tl.load(F + b * N * M + s * M + j, s < N, other=0)
        z = tl.load(F + b * N * M + t * M + j, t < N, other=0)
        le = le & (a[None, :] <= z[:, None])
        lt = lt | (a[None, :] < z[:, None])
    d = le & lt
    if HAS_CV:
        a = tl.load(CV + b * N + s, s < N, other=0)
        z = tl.load(CV + b * N + t, t < N, other=0)
        af, zf = a <= 0, z <= 0
        d = (af[None, :] & ~zf[:, None]) | (
            ~af[None, :] & ~zf[:, None] & (a[None, :] < z[:, None])) | (
            ((af[None, :] & zf[:, None]) | (~af[None, :] & ~zf[:, None] & (a[None, :] == z[:, None]))) & d)
    d = d & (s[None, :] < N) & (t[:, None] < N)
    bits = d.to(tl.uint32) << (s[None, :] % 32)
    words = tl.sum(tl.reshape(bits, (BT, BS // 32, 32)), 2).to(tl.uint32)
    w = tl.program_id(0) * (BS // 32) + tl.arange(0, BS // 32)
    tl.store(D + (b * N + t[:, None]) * W + w[None, :], words, (t[:, None] < N) & (w[None, :] < W))


@triton.jit
def _init(D, Count, Rank, Front, N: tl.constexpr, W: tl.constexpr, BW: tl.constexpr, BT: tl.constexpr):
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    w = tl.arange(0, BW)
    b = tl.program_id(1)
    d = tl.load(D + (b * N + t[:, None]) * W + w[None, :], (t[:, None] < N) & (w[None, :] < W), 0)
    count = tl.sum(_popcount(d), 1)
    tl.store(Count + b * N + t, count, t < N)
    tl.store(Rank + b * N + t, N, t < N)
    tl.store(Front + b * N + t, count == 0, t < N)


@triton.jit
def _pack_front(Front, Seen, Packed, NextSeen, N: tl.constexpr, W: tl.constexpr, B: tl.constexpr):
    b = tl.program_id(0)
    i = tl.arange(0, B)
    front = tl.load(Front + b * N + i, i < N, other=0)
    bits = front.to(tl.uint32) << (i % 32)
    words = tl.sum(tl.reshape(bits, (B // 32, 32)), 1).to(tl.uint32)
    w = tl.arange(0, B // 32)
    tl.store(Packed + b * W + w, words, w < W)
    tl.store(NextSeen + b, tl.load(Seen + b) + tl.sum(front.to(tl.int32), 0))


@triton.jit
def _advance(D, Count, Rank, Front, Packed, Seen, Target, Level, NewCount, NewRank, NewFront,
             N: tl.constexpr, W: tl.constexpr, BW: tl.constexpr, BT: tl.constexpr):
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    w = tl.arange(0, BW)
    b = tl.program_id(1)
    d = tl.load(D + (b * N + t[:, None]) * W + w[None, :], (t[:, None] < N) & (w[None, :] < W), 0)
    pf = tl.load(Packed + b * W + w, w < W, 0)
    decrement = tl.sum(_popcount(d & pf[None, :]), 1)
    front = tl.load(Front + b * N + t, t < N, 0).to(tl.int1)
    count = tl.load(Count + b * N + t, t < N, 0) - decrement - front.to(tl.int32)
    rank = tl.load(Rank + b * N + t, t < N, 0)
    rank = tl.where(front, tl.load(Level), rank)
    active = tl.load(Seen + b) < tl.load(Target + b)
    tl.store(NewCount + b * N + t, count, t < N)
    tl.store(NewRank + b * N + t, rank, t < N)
    tl.store(NewFront + b * N + t, (count == 0) & active, t < N)


def _build_fake(f: torch.Tensor, cv: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = f.shape[0]
    shape = (n,)
    return (torch.empty((*shape, triton.cdiv(n, 32)), device=f.device, dtype=torch.uint32),
            torch.empty(shape, device=f.device, dtype=torch.int32),
            torch.empty(shape, device=f.device, dtype=torch.int32),
            torch.empty(shape, device=f.device, dtype=torch.bool))


def _build_impl(f: torch.Tensor, cv: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    f = f.contiguous()
    n, m = f.shape
    if cv is not None:
        cv = cv.contiguous()
    d, count, rank, front = _build_fake(f, cv)
    w = triton.cdiv(n, 32)
    _pack_dom[(triton.cdiv(n, 128), triton.cdiv(n, 16), 1)](f, cv, d, n, m, w, cv is not None, 128, 16)
    _init[(triton.cdiv(n, 16), 1)](d, count, rank, front, n, w, triton.next_power_of_2(w), 16)
    return d, count, rank, front


_build = torch.library.custom_op("evomo_triton_nd::build", _build_impl, mutates_args=())
_build.register_fake(_build_fake)


def _step_fake(d: torch.Tensor, count: torch.Tensor, rank: torch.Tensor, front: torch.Tensor,
               seen: torch.Tensor, target: torch.Tensor, level: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.empty_like(count), torch.empty_like(rank), torch.empty_like(front), torch.empty_like(seen)


def _step_impl(d: torch.Tensor, count: torch.Tensor, rank: torch.Tensor, front: torch.Tensor,
               seen: torch.Tensor, target: torch.Tensor, level: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n, w = d.shape
    new_count, new_rank, new_front, new_seen = _step_fake(d, count, rank, front, seen, target, level)
    packed = torch.empty((w,), device=d.device, dtype=torch.uint32)
    torch.library.wrap_triton(_pack_front)[(1,)](front, seen, packed, new_seen, n, w, max(32, triton.next_power_of_2(n)))
    torch.library.wrap_triton(_advance)[(triton.cdiv(n, 16), 1)](
        d, count, rank, front, packed, new_seen, target, level,
        new_count, new_rank, new_front, n, w, triton.next_power_of_2(w), 16)
    return new_count, new_rank, new_front, new_seen


# Make per-front kernel launches visible to Inductor instead of entering an
# opaque Python custom-op body on every loop iteration.
_step = torch.library.triton_op("evomo_triton_nd::step", _step_impl, mutates_args=())


def _rank_loop(d, count, rank, front, target):
    seen = torch.zeros((), dtype=torch.int32, device=d.device)

    def cond(count, rank, front, seen, level):
        return front.any()

    def body(count, rank, front, seen, level):
        count, rank, front, seen = _step(d, count, rank, front, seen, target, level)
        return count, rank, front, seen, level + 1

    _, rank, *_ = torch.while_loop(cond, body, (count, rank, front, seen, torch.zeros((), device=d.device, dtype=torch.int32)))
    return rank


_compiled_rank_loop = torch.compile(_rank_loop, fullgraph=True)


def _rank_fake(d: torch.Tensor, count: torch.Tensor, rank: torch.Tensor, front: torch.Tensor,
               target: torch.Tensor, compiling: bool) -> torch.Tensor:
    return torch.empty_like(count)


def _rank_impl(d: torch.Tensor, count: torch.Tensor, rank: torch.Tensor, front: torch.Tensor,
               target: torch.Tensor, compiling: bool) -> torch.Tensor:
    if compiling:
        return _compiled_rank_loop(d, count, rank, front, target)
    seen = torch.zeros((), dtype=torch.int32, device=d.device)
    level = torch.zeros((), dtype=torch.int32, device=d.device)
    while front.any():
        count, rank, front, seen = _step_impl(d, count, rank, front, seen, target, level)
        level += 1
    return rank


_rank = torch.library.custom_op("evomo_triton_nd::rank", _rank_impl, mutates_args=())
_rank.register_fake(_rank_fake)


def _ranks(f, cv, topk):
    _check_input(f, cv)
    if f.device.type != "cuda":
        return reference._environmental_selection_rank(f, topk, cv)
    cv_sum = None if cv is None else (cv.sum(dim=1) if cv.ndim > 1 else cv)
    target = torch.tensor(topk, device=f.device, dtype=torch.int32)
    if cv_sum is not None:
        target = torch.where(((cv_sum < 0) | ~torch.isfinite(cv_sum)).any(), f.shape[0], target)
    d, count, rank, front = _build(f, cv_sum)
    return _rank(d, count, rank, front, target, torch.compiler.is_compiling())


def non_dominate_rank(f, cv=None):
    """Return complete zero-based ranks, including constrained dominance."""
    _check_input(f, cv)
    if f.shape[0] == 0:
        return reference.non_dominate_rank(f, cv)
    return _ranks(f, cv, f.shape[0])


@triton.jit
def _dense_relation(X, Y, CX, CY, Out, NX: tl.constexpr, NY: tl.constexpr, M: tl.constexpr,
                    CV: tl.constexpr, B: tl.constexpr):
    p = tl.program_id(0) * B + tl.arange(0, B)
    batch = tl.program_id(1)
    i, j = p // NY, p % NY
    valid = p < NX * NY
    le = tl.full((B,), True, tl.int1)
    lt = tl.full((B,), False, tl.int1)
    for k in range(M):
        x = tl.load(X + (batch * NX + i) * M + k, valid, 0)
        y = tl.load(Y + (batch * NY + j) * M + k, valid, 0)
        le, lt = le & (x <= y), lt | (x < y)
    d = le & lt
    if CV:
        x = tl.load(CX + batch * NX + i, valid, 0)
        y = tl.load(CY + batch * NY + j, valid, 0)
        xf, yf = x <= 0, y <= 0
        d = (xf & ~yf) | (~xf & ~yf & (x < y)) | (((xf & yf) | (~xf & ~yf & (x == y))) & d)
    tl.store(Out + batch * NX * NY + p, d, valid)


def _dense_fake(x: torch.Tensor, y: torch.Tensor, cx: torch.Tensor | None, cy: torch.Tensor | None) -> torch.Tensor:
    return torch.empty((x.shape[0], y.shape[0]), device=x.device, dtype=torch.bool)


def _dense_impl(x: torch.Tensor, y: torch.Tensor, cx: torch.Tensor | None, cy: torch.Tensor | None) -> torch.Tensor:
    result = _dense_fake(x, y, cx, cy)
    x, y = x.contiguous(), y.contiguous()
    cx = None if cx is None else cx.contiguous()
    cy = None if cy is None else cy.contiguous()
    nx, ny = result.shape
    _dense_relation[(triton.cdiv(nx * ny, 256), 1)](x, y, cx, cy, result, nx, ny, x.shape[1],
                                                               cx is not None and cy is not None, 256)
    return result


_dense = torch.library.custom_op("evomo_triton_nd::dense", _dense_impl, mutates_args=())
_dense.register_fake(_dense_fake)


def dominate_relation(x, y, cv_x=None, cv_y=None):
    """Return the dense matrix whose (i, j) entry means x[i] dominates y[j]."""
    _check_input(x, cv_x)
    _check_input(y, cv_y)
    if x.shape[1] != y.shape[1] or x.device != y.device:
        raise ValueError("Both objective arrays must have matching objective counts and devices.")
    if x.device.type != "cuda" or x.shape[0] == 0 or y.shape[0] == 0:
        return reference.dominate_relation(x, y, cv_x, cv_y)
    cx = None if cv_x is None else (cv_x.sum(dim=1) if cv_x.ndim > 1 else cv_x)
    cy = None if cv_y is None else (cv_y.sum(dim=1) if cv_y.ndim > 1 else cv_y)
    return _dense(x, y, cx, cy)


def nd_environmental_selection(x, f, topk, cv=None):
    """Packed Triton ranking plus unchanged PyTorch crowding/stable selection."""
    rank = _ranks(f, cv, topk)
    cutoff = torch.topk(rank, topk, largest=False).values[-1]
    distance = reference.crowding_distance(f, rank == cutoff)
    keys = [-distance, rank]
    if cv is not None:
        keys.append(cv.sum(dim=1) if cv.ndim > 1 else cv)
    order = lexsort(keys)[:topk]
    return x[order], f[order], rank[order], distance[order], None if cv is None else cv[order]


# Both backends expose the same operator interface.
crowding_distance = reference.crowding_distance
