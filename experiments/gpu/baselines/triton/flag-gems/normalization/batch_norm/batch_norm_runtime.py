import importlib.util
import sys
from pathlib import Path

import torch


SOURCE = Path(__file__).with_name("batch_norm.py")
SPEC = importlib.util.spec_from_file_location("intent_flaggems_batch_norm", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
STATE = {}


def upstream(arguments):
    x, weight, bias, running_mean, running_var, epsilon, momentum = arguments
    key = (
        tuple(x.shape),
        x.dtype,
        x.device,
        weight.dtype,
        bias.dtype,
    )
    if key not in STATE:
        channels = x.shape[1]
        STATE[key] = (
            torch.empty_like(x),
            torch.empty((channels,), device=x.device, dtype=torch.float32),
            torch.empty((channels,), device=x.device, dtype=torch.float32),
        )
    output, mean, inverse_std = STATE[key]
    MODULE.batch_norm_forward_kernel[(x.shape[1],)](
        x,
        weight,
        bias,
        mean,
        inverse_std,
        output,
        running_mean,
        running_var,
        x.shape[0],
        x.shape[2],
        *x.stride(),
        *output.stride(),
        momentum,
        epsilon,
        is_train=True,
    )
    return output, mean, inverse_std
