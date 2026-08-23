from __future__ import annotations

from .attention import CASES as ATTENTION_CASES
from .contraction import CASES as CONTRACTION_CASES
from .convolution import CASES as CONVOLUTION_CASES
from .factorization import CASES as FACTORIZATION_CASES
from .indexing import CASES as INDEXING_CASES
from .loss import CASES as LOSS_CASES
from .mamba import CASES as MAMBA_CASES
from .normalization import CASES as NORMALIZATION_CASES
from .optimization import CASES as OPTIMIZATION_CASES
from .pointwise import CASES as POINTWISE_CASES
from .position import CASES as POSITION_CASES
from .quantization import CASES as QUANTIZATION_CASES
from .ragged import CASES as RAGGED_CASES
from .scan import CASES as SCAN_CASES
from .statistics import CASES as STATISTICS_CASES
from .vision import CASES as VISION_CASES


CASES = {
    **ATTENTION_CASES,
    **CONTRACTION_CASES,
    **CONVOLUTION_CASES,
    **FACTORIZATION_CASES,
    **INDEXING_CASES,
    **LOSS_CASES,
    **MAMBA_CASES,
    **NORMALIZATION_CASES,
    **OPTIMIZATION_CASES,
    **POINTWISE_CASES,
    **POSITION_CASES,
    **QUANTIZATION_CASES,
    **RAGGED_CASES,
    **SCAN_CASES,
    **STATISTICS_CASES,
    **VISION_CASES,
}
