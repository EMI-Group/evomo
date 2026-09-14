"""Resolve an optional backend before compiling an algorithm or operator."""

from importlib import import_module
from types import ModuleType


def get_non_dominate_backend(backend: str = "torch") -> ModuleType:
    """Return the ``torch`` (default) or experimental ``triton`` operator module.

    Call this during setup, outside ``torch.compile``. Triton is imported only
    when explicitly requested. The returned module exposes non_dominate_rank,
    dominate_relation, crowding_distance and nd_environmental_selection.
    """
    if backend == "torch":
        return import_module(".non_dominate", __package__)
    if backend != "triton":
        raise ValueError(f"Unknown non-dominated sorting backend {backend!r}; expected 'torch' or 'triton'.")
    try:
        return import_module(".non_dominate_triton", __package__)
    except ModuleNotFoundError as exc:
        if exc.name == "triton":
            raise ImportError(
                "The Triton backend requires a Triton installation compatible with your PyTorch and platform. "
                "Use backend='torch' without this optional dependency."
            ) from exc
        raise
