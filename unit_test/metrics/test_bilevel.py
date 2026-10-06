"""Bilevel distance metrics, Monte Carlo hypervolume and tensor execution."""

import math

import pytest
import torch

from evomo.metrics import gd, igd, lgd, uhv, uigd

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bilevel_distance_metrics(device, dtype):
    points = torch.tensor([[3.0, 4.0], [0.0, 12.0]], device=device, dtype=dtype)
    front = points.new_zeros(1, 2)
    torch.testing.assert_close(lgd(points, front), points.new_tensor(6.5))
    torch.testing.assert_close(lgd(points, front), gd(points, front))
    front = points.new_tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    points = points.new_zeros(1, 2)
    torch.testing.assert_close(uigd(points, front), points.new_tensor(4 / 3))
    torch.testing.assert_close(uigd(points, front, p=2), points.new_tensor(math.sqrt(8 / 3)))
    torch.testing.assert_close(uigd(points, front), igd(points, front))
    assert lgd(points, front).dtype == dtype
    assert uigd(points, front).device.type == device


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_uhv(device, dtype):
    points = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device, dtype=dtype)
    ref = points.new_tensor([2.0, 2.0])
    grid = (torch.arange(4, device=device, dtype=dtype) + 0.5) / 4
    samples = torch.cartesian_prod(grid, grid)
    value = uhv(points, ref, 16, samples=samples, chunk_size=3, point_chunk_size=1)
    torch.testing.assert_close(value, points.new_tensor(3.0))
    torch.testing.assert_close(uhv(points, ref, 16, samples=samples, chunk_size=16), value)
    # Duplicate, dominated, nonfinite and outside-reference points add no volume.
    extra = points.new_tensor([[0.0, 1.0], [1.5, 1.5], [-10.0, 3.0], [float("nan"), 0.0], [0.0, float("inf")]])
    torch.testing.assert_close(uhv(torch.cat((points, extra)), ref, 16, samples=samples, point_chunk_size=2), value)
    mask = torch.tensor([True, False], device=device)
    torch.testing.assert_close(uhv(points, ref, 16, mask=mask, samples=samples), points.new_tensor(2.0))
    assert value.dtype == dtype and value.device.type == device


def test_uhv_monte_carlo():
    points = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device="cpu", dtype=torch.float64)
    ref = points.new_tensor([2.0, 2.0])
    # Compare multiple seeds with the analytic union area 3, not another estimator.
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        value = uhv(points, ref, 50000)
        assert abs(float(value) - 3.0) < 0.035
    torch.manual_seed(42)
    first = uhv(points, ref, 1003, chunk_size=300)
    torch.manual_seed(42)
    second = uhv(points, ref, 1003, chunk_size=300)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def metric_bundle(points, lower_front, upper_front, ref, mask, samples):
    return (
        lgd(points, lower_front, mask=mask),
        uigd(points, upper_front, mask=mask),
        uhv(points, ref, samples.shape[-2], mask=mask, samples=samples, chunk_size=17, point_chunk_size=1),
    )


def bundle_inputs(device, dtype):
    points = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.5, 1.5]], device=device, dtype=dtype)
    lower_front = points.new_zeros(3, 2, 2)
    upper_front = points.new_tensor([[0.0, 0.0], [0.0, 2.0]])
    ref = points.new_tensor([2.0, 2.0])
    mask = torch.tensor([True, True, False], device=device)
    samples = torch.rand(64, 2, device=device, dtype=dtype)
    return points, lower_front, upper_front, ref, mask, samples


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bilevel_metrics_vmap(device, dtype):
    inputs = bundle_inputs(device, dtype)
    batched = tuple(torch.stack((v, v)) for v in inputs)
    expected = tuple(torch.stack((v, v)) for v in metric_bundle(*inputs))
    for result in (metric_bundle(*batched), torch.vmap(metric_bundle)(*batched)):
        for actual, target in zip(result, expected):
            torch.testing.assert_close(actual, target)
    # Shared references and samples also broadcast across metric batches.
    points, lower_front, upper_front, ref, mask, samples = inputs
    torch.testing.assert_close(uigd(batched[0], upper_front, mask=batched[4]), expected[1])
    torch.testing.assert_close(uhv(batched[0], ref, 64, samples=samples, mask=batched[4]), expected[2])
    mapped_random = torch.vmap(lambda x: uhv(x, ref, 64), randomness="different")(batched[0])
    assert mapped_random.shape == (2,) and torch.isfinite(mapped_random).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bilevel_metrics_compiled(device, dtype):
    inputs = bundle_inputs(device, dtype)
    compiled = torch.compile(metric_bundle, backend="eager", fullgraph=True)
    for actual, expected in zip(compiled(*inputs), metric_bundle(*inputs)):
        torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bilevel_metrics_cuda_inductor(dtype):
    inputs = bundle_inputs("cuda", dtype)
    compiled = torch.compile(metric_bundle, fullgraph=True)
    for actual, expected in zip(compiled(*inputs), metric_bundle(*inputs)):
        torch.testing.assert_close(actual, expected)
    # Internal random sampling is compiled too; this single-box volume is exact.
    point, ref = inputs[0][:1], inputs[3]
    random_compiled = torch.compile(lambda x, r: uhv(x, r, 100, chunk_size=31), fullgraph=True)
    torch.testing.assert_close(random_compiled(point, ref), point.new_tensor(2.0))
