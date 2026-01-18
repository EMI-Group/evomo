__all__ = ["neuroevolution", "numerical"]


from . import numerical

try:
    from . import neuroevolution
except ModuleNotFoundError:
    neuroevolution = None  # type: ignore
