from __future__ import annotations

from .attention import CASES as ATTENTION_CASES
from .contraction import CASES as CONTRACTION_CASES
from .convolution import CASES as CONVOLUTION_CASES
from .indexing import CASES as INDEXING_CASES
from .loss import CASES as LOSS_CASES
from .mamba import CASES as MAMBA_CASES
from .normalization import CASES as NORMALIZATION_CASES
from .position import CASES as POSITION_CASES
from .quantization import CASES as QUANTIZATION_CASES
from .ragged import CASES as RAGGED_CASES


CASES = {
    **ATTENTION_CASES,
    **CONTRACTION_CASES,
    **CONVOLUTION_CASES,
    **INDEXING_CASES,
    **LOSS_CASES,
    **MAMBA_CASES,
    **NORMALIZATION_CASES,
    **POSITION_CASES,
    **QUANTIZATION_CASES,
    **RAGGED_CASES,
}
