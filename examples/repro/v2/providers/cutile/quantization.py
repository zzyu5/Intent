from __future__ import annotations

from .. import implementation_gap


CASES = {
    "nvfp4_quantize": implementation_gap(
        "the algorithm requires typed FP4 packing plus the swizzled FP8 scale "
        "storage contract, which the current Intent Kernel IR cannot express"
    ),
}
