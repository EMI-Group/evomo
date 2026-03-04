import os
import numpy as np
from typing import Union, Tuple
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
    # Get the path to src/evomo/problems/constrained/pf
    # __file__ is src/evomo/utils/common.py
    # utils -> evomo -> problems -> constrained -> pf
    utils_dir = os.path.dirname(os.path.realpath(__file__))
    evomo_dir = os.path.dirname(utils_dir)
    pf_dir = os.path.join(evomo_dir, "problems", "constrained", "pf")
    
    full_path = os.path.join(pf_dir, fname)

    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"Pareto front file not found: {full_path}")

    # load using numpy
    pf_np = np.loadtxt(full_path)

    # convert to torch tensor
    pf = torch.as_tensor(pf_np, dtype=dtype)

    # sort by first column
    pf = pf[pf[:, 0].argsort()]

    # move to target device if specified
    if device is not None:
        pf = pf.to(device)

    return pf
