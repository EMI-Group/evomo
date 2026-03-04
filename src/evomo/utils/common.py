from typing import Union, Tuple
import torch

def parse_evaluate(eval_out: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
    """Parse output from evaluate function. Returns (fit, cv) where cv is None if unconstrained."""
    if isinstance(eval_out, tuple):
        return eval_out[0], eval_out[1]
    return eval_out, None
