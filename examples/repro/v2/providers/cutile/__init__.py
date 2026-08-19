from __future__ import annotations

from .activation import CASES as ACTIVATION_CASES
from .attention import CASES as ATTENTION_CASES
from .contraction import CASES as CONTRACTION_CASES
from .normalization import CASES as NORMALIZATION_CASES
from .moe import CASES as MOE_CASES
from .position import CASES as POSITION_CASES
from .regularization import CASES as REGULARIZATION_CASES
from .routing import CASES as ROUTING_CASES
from .scan import CASES as SCAN_CASES


CASES = {
    **ACTIVATION_CASES,
    **ATTENTION_CASES,
    **CONTRACTION_CASES,
    **NORMALIZATION_CASES,
    **MOE_CASES,
    **POSITION_CASES,
    **REGULARIZATION_CASES,
    **ROUTING_CASES,
    **SCAN_CASES,
}
