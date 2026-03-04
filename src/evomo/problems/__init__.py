__all__ = ["constrained", "neuroevolution", "numerical"]


from . import numerical
from . import constrained
try:
    from . import neuroevolution
except ModuleNotFoundError:
    neuroevolution = None  # type: ignore
