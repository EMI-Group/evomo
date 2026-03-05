__all__ = ["constrained", "neuroevolution", "numerical"]


from . import constrained, numerical

try:
    from . import neuroevolution
except ModuleNotFoundError:
    neuroevolution = None  # type: ignore
