"""Core hypervolume correctness, numeric and execution contracts."""

import itertools
import math

import pytest
import torch

from evomo.metrics import hv

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def exact_union(points, reference):
    """Independent inclusion-exclusion oracle for small rectangle unions."""
    valid = [p for p in points if all(math.isfinite(x) and x < r for x, r in zip(p, reference))]
    volume = 0.0
    for count in range(1, len(valid) + 1):
        for subset in itertools.combinations(valid, count):
            intersection = math.prod(max(0.0, r - max(p[j] for p in subset)) for j, r in enumerate(reference))
            volume += (-1) ** (count + 1) * intersection
    return volume


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_single_box_empty_set_and_numeric_dtype(device, dtype):
    points = torch.zeros(1, 3, device=device, dtype=dtype)
    reference = torch.full((3,), 100, device=device, dtype=dtype)
    result = hv(points, reference, 128)
    assert result.device == points.device and result.dtype == torch.float32
    assert result.item() == 1_000_000
    empty = hv(points[:0], reference, 128)
    assert empty.shape == () and empty.dtype == torch.float32 and empty.device == points.device
    assert empty.item() == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "points,reference",
    [
        ([[0.2, 0.8], [0.5, 0.4], [0.8, 0.2]], [1.0, 1.0]),
        ([[1.0, 0.5, 1.5], [0.5, 1.5, 1.0]], [2.0, 2.0, 2.0]),
        ([[-1.0, 0.5], [0.0, -0.5], [-1.0, 0.5], [0.5, 0.8]], [1.0, 1.0]),
        ([[0.1, 0.7, 0.4, 0.6, 0.2], [0.6, 0.2, 0.5, 0.3, 0.4]], [1.0] * 5),
    ],
)
def test_monte_carlo_matches_independent_union(device, points, reference):
    objectives = torch.tensor(points, device=device, dtype=torch.float64)
    ref = objectives.new_tensor(reference)
    expected = exact_union(points, reference)
    bounding_volume = math.prod(r - min(p[j] for p in points) for j, r in enumerate(reference))
    samples = 100_000
    probability = expected / bounding_volume
    tolerance = 6 * bounding_volume * math.sqrt(probability * (1 - probability) / samples)
    for seed in (1, 7, 23):
        torch.manual_seed(seed)
        assert abs(hv(objectives, ref, samples).item() - expected) < tolerance


@pytest.mark.parametrize("device", DEVICES)
def test_invalid_outside_and_zero_volume_points_do_not_change_valid_box(device):
    points = torch.tensor(
        [[0.5, 0.5], [0.75, 0.75], [float("nan"), 0], [float("inf"), 0], [float("-inf"), 0], [-100, 1], [-100, 2]],
        device=device,
    )
    assert hv(points, points.new_ones(2), 256).item() == 0.25
    assert hv(points[2:], points.new_ones(2), 256).item() == 0
    assert hv(points.new_ones((3, 2)), points.new_ones(2), 256).item() == 0


@pytest.mark.parametrize("device", DEVICES)
def test_integer_subtraction_does_not_overflow_before_float_conversion(device):
    points = torch.tensor([[-(2**63)]], dtype=torch.int64, device=device)
    reference = torch.ones(1, dtype=torch.int64, device=device)
    torch.testing.assert_close(hv(points, reference, 128), torch.tensor(float(2**63), device=device))


@pytest.mark.parametrize("device", DEVICES)
def test_fullgraph_eager_and_compiled_execution(device):
    torch.compiler.reset()
    backend = "aot_eager" if device == "cpu" else "inductor"
    function = torch.compile(hv, fullgraph=True, backend=backend)
    points = torch.tensor([[0.5, 0.5], [0.75, 0.75]], device=device)
    reference = points.new_ones(2)
    torch.testing.assert_close(function(points, reference, 256), hv(points, reference, 256))
    assert function(points[:0], reference, 256).item() == 0
    torch.compiler.reset()
