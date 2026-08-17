import importlib.util
import sys
from pathlib import Path

import torch


SOURCE = Path(__file__).with_name("softmax.py")
SPEC = importlib.util.spec_from_file_location("intent_flaggems_softmax", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
STATE = {}


def upstream(arguments):
    probabilities, gradient = arguments
    key = (tuple(probabilities.shape), probabilities.dtype)
    if key not in STATE:
        STATE[key] = torch.empty_like(probabilities)
    return MODULE.softmax_backward_out(
        gradient,
        probabilities,
        dim=1,
        input_dtype=torch.float32,
        grad_input=STATE[key],
    )
