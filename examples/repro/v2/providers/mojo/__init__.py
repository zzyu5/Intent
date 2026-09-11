from .pointwise import CASES as POINTWISE
from .normalization import CASES as NORMALIZATION
from .contraction import CASES as CONTRACTION
from .attention import CASES as ATTENTION

CASES = {**POINTWISE, **NORMALIZATION, **CONTRACTION, **ATTENTION}
