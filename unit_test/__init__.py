"""Shared unittest runtime isolation for tensor tests."""

from contextlib import contextmanager
from unittest import TestCase

import torch


class TorchTestCase(TestCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(torch.set_default_device, torch.get_default_device())
        self.addCleanup(torch.compiler.reset)
        torch.set_default_device("cpu")
        torch.compiler.reset()

    @contextmanager
    def torch_case(self, **parameters):
        """Keep each parameter combination independent, including failed subtests."""
        with self.subTest(**parameters):
            device = torch.get_default_device()
            try:
                torch.set_default_device("cpu")
                torch.compiler.reset()
                yield
            finally:
                torch.compiler.reset()
                torch.set_default_device(device)
