import importlib.util
import sys
from pathlib import Path

import torch


SOURCE = Path(__file__).with_name("fp8_mqa_logits.py")
SPEC = importlib.util.spec_from_file_location("intent_flaggems_fp8_mqa_logits", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
STATE = {}


def upstream(arguments):
    q, kv, kv_scale, head_weight, key_start, key_end = arguments
    key = (tuple(q.shape), tuple(kv.shape))
    if key not in STATE:
        STATE[key] = torch.empty(
            (q.shape[0], kv.shape[0]), device=q.device, dtype=torch.float32
        )
    output = STATE[key]
    grid = lambda meta: (
        MODULE.triton.cdiv(q.shape[0], meta["BLOCK_M"]),
        MODULE.triton.cdiv(kv.shape[0], meta["BLOCK_N"]),
    )
    MODULE._fp8_mqa_logits_kernel[grid](
        q,
        kv,
        kv_scale,
        head_weight,
        key_start,
        key_end,
        output,
        *q.stride(),
        *kv.stride(),
        q.shape[0],
        q.shape[1],
        q.shape[2],
        kv.shape[0],
        True,
    )
    return output
