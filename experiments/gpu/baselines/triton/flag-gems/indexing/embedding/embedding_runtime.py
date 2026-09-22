import importlib.util
import sys
from pathlib import Path

import torch


SOURCE = Path(__file__).with_name("embedding.py")
SPEC = importlib.util.spec_from_file_location("intent_flaggems_embedding", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
STATE = {}


def upstream(arguments):
    labels, table = arguments
    key = (tuple(labels.shape), tuple(table.shape), table.dtype)
    if key not in STATE:
        STATE[key] = torch.empty(
            (*labels.shape, table.shape[-1]),
            device=table.device,
            dtype=table.dtype,
        )
    output = STATE[key]
    block = MODULE.triton.next_power_of_2(table.shape[-1])
    MODULE.embedding_kernel[(labels.numel(),)](
        output,
        labels,
        table,
        table.shape[-1],
        block,
    )
    return output
