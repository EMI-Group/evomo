"""Helpers for optional algorithm state whose shape is known after evaluation."""

from functools import partial

import torch


def _materialize_lazy_buffer(
    module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs, *, name, device_like
):
    saved = state_dict.get(prefix + name)
    if module.get_buffer(name) is None and isinstance(saved, torch.Tensor):
        reference = getattr(module, device_like)
        setattr(module, name, torch.empty_like(saved, device=reference.device))


def register_lazy_buffer(module: torch.nn.Module, name: str, *, device_like: str) -> None:
    """Register optional persistent state, materializing it before checkpoint loading.

    The buffer starts as ``None`` and accepts normal tensor assignments after
    evaluation. A fresh instance can restore its shape and dtype from a checkpoint.
    Already initialized buffers retain PyTorch's normal shape validation.

    ``device_like`` names an existing tensor attribute on the same module, such as
    ``"pop"``. Its device is read at load time, so use a parameter or buffer that
    follows module device transfers. The tensor itself is not captured by the hook.
    Standard ``load_state_dict(assign=True)`` semantics still take precedence:
    assigned tensors retain the checkpoint's device and dtype.

    As with ordinary None buffers, absent checkpoint keys leave fresh state as None.
    This helper does not infer or validate a problem's number of constraints.

    Example::

        self.pop = Mutable(population)
        register_lazy_buffer(self, "cv", device_like="pop")
    """
    if name == device_like or not isinstance(getattr(module, device_like, None), torch.Tensor):
        raise ValueError("device_like must name an existing, distinct tensor attribute on the module")
    if hasattr(module, name):
        raise ValueError(f"Attribute {name!r} already exists; register lazy buffers only once")
    module.register_buffer(name, None)
    module.register_load_state_dict_pre_hook(partial(_materialize_lazy_buffer, name=name, device_like=device_like))
