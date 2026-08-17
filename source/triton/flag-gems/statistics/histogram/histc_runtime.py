import importlib.util
import sys
from pathlib import Path

import torch


SOURCE = Path(__file__).with_name("histc.py")
SPEC = importlib.util.spec_from_file_location("intent_flaggems_histc", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
STATE = {}


def upstream(arguments):
    samples = arguments[0]
    key = (tuple(samples.shape), samples.dtype, samples.device)
    if key not in STATE:
        STATE[key] = torch.zeros((256,), device=samples.device, dtype=samples.dtype)
    output = STATE[key]
    output.zero_()
    grid = (MODULE.triton.cdiv(samples.numel(), 1024),)
    MODULE.histc_kernel_simple[grid](
        samples,
        output,
        samples.numel(),
        256,
        0.0,
        256.0,
        BLOCK_SIZE=1024,
    )
    return output
