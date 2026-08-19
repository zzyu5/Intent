from __future__ import annotations

from .contraction import CASES as CONTRACTION_CASES
from .convolution import CASES as CONVOLUTION_CASES
from .normalization import CASES as NORMALIZATION_CASES
from .quantization import CASES as QUANTIZATION_CASES
from .routing import CASES as ROUTING_CASES
from .streaming import CASES as STREAMING_CASES
from .attention import CASES as ATTENTION_CASES


CASES = {
    **CONTRACTION_CASES,
    **CONVOLUTION_CASES,
    **NORMALIZATION_CASES,
    **QUANTIZATION_CASES,
    **ROUTING_CASES,
    **STREAMING_CASES,
    **ATTENTION_CASES,
}
