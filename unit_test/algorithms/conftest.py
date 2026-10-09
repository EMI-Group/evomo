"""Keep algorithm tests independent of global device and compiler state."""

import pytest
import torch


@pytest.fixture(autouse=True)
def isolate_torch_runtime():
    device = torch.get_default_device()
    torch.set_default_device("cpu")
    torch.compiler.reset()
    yield
    torch.compiler.reset()
    torch.set_default_device(device)
