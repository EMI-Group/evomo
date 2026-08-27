from importlib.resources import files
from pathlib import PurePosixPath
from typing import Tuple, Union

import numpy as np
import torch


def parse_evaluate(eval_out: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
    """Parse output from evaluate function. Returns (fit, cv) where cv is None if unconstrained."""
    if isinstance(eval_out, tuple):
        return eval_out[0], eval_out[1]
    return eval_out, None


def load_pareto_front_from_file(fname, device=None, dtype=torch.float32):
    """
    Load Pareto front from the constrained problems' pf directory.
    """
    resource = files("evomo.problems.constrained").joinpath("pf", *PurePosixPath(fname).parts)

    if not resource.is_file():
        raise FileNotFoundError(f"Pareto front resource not found: {fname}")

    # load using numpy
    with resource.open("r", encoding="utf-8") as file:
        pf_np = np.loadtxt(file)

    # convert to torch tensor
    pf = torch.as_tensor(pf_np, dtype=dtype)

    # sort by first column
    pf = pf[pf[:, 0].argsort()]

    # move to target device if specified
    if device is not None:
        pf = pf.to(device)

    return pf
