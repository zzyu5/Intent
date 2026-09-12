from .pointwise import CASES as POINTWISE
from .normalization import CASES as NORMALIZATION
from .contraction import CASES as CONTRACTION
from .attention import CASES as ATTENTION
from .activation import CASES as ACTIVATION
from .layout import CASES as LAYOUT
from .optimization import CASES as OPTIMIZATION
from .indexing import CASES as INDEXING
from .sparse import CASES as SPARSE
from .vision import CASES as VISION
from .convolution import CASES as CONVOLUTION
from .factorization import CASES as FACTORIZATION
from .sorting import CASES as SORTING
from .dynamic_programming import CASES as DYNAMIC_PROGRAMMING
from .spectral import CASES as SPECTRAL
from .scan import CASES as SCAN
from .statistics import CASES as STATISTICS
from .position import CASES as POSITION
from .atomics import CASES as ATOMICS
from .loss import CASES as LOSS
from .backward import CASES as BACKWARD

CASES = {**POINTWISE, **NORMALIZATION, **CONTRACTION, **ATTENTION, **ACTIVATION, **LAYOUT, **OPTIMIZATION, **INDEXING, **SPARSE, **VISION, **POSITION, **CONVOLUTION, **FACTORIZATION, **SORTING, **DYNAMIC_PROGRAMMING, **SPECTRAL, **SCAN, **STATISTICS, **ATOMICS, **LOSS, **BACKWARD}
