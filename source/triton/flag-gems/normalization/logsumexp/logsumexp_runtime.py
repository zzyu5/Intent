import importlib.util
import sys
from pathlib import Path

import torch


SOURCE = Path(__file__).with_name("logsumexp.py")
SPEC = importlib.util.spec_from_file_location("intent_flaggems_logsumexp", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
STATE = {}


def upstream(arguments):
    x = arguments[0]
    key = (tuple(x.shape), x.dtype)
    if key not in STATE:
        STATE[key] = torch.empty((x.shape[0],), device=x.device, dtype=x.dtype)
    output = STATE[key]
    MODULE.logsumexp_kernel_inner[(x.shape[0], 1, 1)](
        output,
        x,
        x.shape[0],
        x.shape[1],
    )
    return output
