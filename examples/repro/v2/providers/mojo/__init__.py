from .pointwise import CASES as POINTWISE
from .normalization import CASES as NORMALIZATION
from .contraction import CASES as CONTRACTION
from .attention import CASES as ATTENTION
from .activation import CASES as ACTIVATION
from .layout import CASES as LAYOUT
from .optimization import CASES as OPTIMIZATION
from .indexing import CASES as INDEXING

CASES = {**POINTWISE, **NORMALIZATION, **CONTRACTION, **ATTENTION, **ACTIVATION, **LAYOUT, **OPTIMIZATION, **INDEXING}
