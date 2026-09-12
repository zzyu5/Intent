from .pointwise import CASES as POINTWISE
from .normalization import CASES as NORMALIZATION
from .contraction import CASES as CONTRACTION
from .attention import CASES as ATTENTION
from .activation import CASES as ACTIVATION
from .layout import CASES as LAYOUT

CASES = {**POINTWISE, **NORMALIZATION, **CONTRACTION, **ATTENTION, **ACTIVATION, **LAYOUT}
