import importlib.util
import sys
from pathlib import Path

import torch


SOURCE = Path(__file__).with_name("cumsum.py")
SPEC = importlib.util.spec_from_file_location("intent_flaggems_cumsum", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
STATE = {}


def upstream(arguments):
    x = arguments[0]
    rows = x.shape[0]
    columns = x.numel() // rows
    key = (tuple(x.shape), x.dtype)
    if key not in STATE:
        STATE[key] = torch.empty_like(x)
    output = STATE[key]
    tile = MODULE.triton.next_power_of_2(columns)
    MODULE.reduce_then_scan_root_scan_kernel_row[(rows, 1, 1)](
        x,
        output,
        columns,
        tile,
        num_warps=4,
    )
    return output
