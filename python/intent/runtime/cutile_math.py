"""Tile implementations of ordinary floating-point library operations."""

# The erf/erfc rational coefficients and intervals are from fdlibm:
# https://github.com/JuliaMath/openlibm/blob/master/src/s_erff.c
# https://github.com/JuliaMath/openlibm/blob/master/src/s_erf.c
# Float conversion by Ian Lance Taylor, Cygnus Support.
# Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
# Developed at SunPro, a Sun Microsystems, Inc. business.
# Permission to use, copy, modify, and distribute this software is freely
# granted, provided that this notice is preserved.

import cuda.tile as ct


@ct.function
def _erf_small(x):
    square = x * x
    numerator = square * -1.86260219e-03 - 3.36030394e-01
    numerator = numerator * square + 1.28379166e-01
    denominator = square * -1.98859419e-03 + 2.16070302e-02
    denominator = denominator * square + 3.12324286e-01
    denominator = denominator * square + 1.0
    value = x + x * (numerator / denominator)
    tiny = (8.0 * x + 1.0270333290 * x) / 8.0
    return ct.where(x < 2.0 ** -119, tiny, value)


@ct.function
def _erf_near_one(x):
    shifted = x - 1.0
    numerator = shifted * 1.10914491e-01 - 1.65179938e-01
    numerator = numerator * shifted + 4.15109694e-01
    numerator = numerator * shifted + 3.64939137e-06
    denominator = shifted * 5.62181212e-02 + 1.68576106e-01
    denominator = denominator * shifted + 5.35934687e-01
    denominator = denominator * shifted + 6.02074385e-01
    denominator = denominator * shifted + 1.0
    return 8.42697144e-01 + numerator / denominator


@ct.function
def _erfc_tail(x):
    inverse_square = 1.0 / (x * x)
    r1 = inverse_square * -1.43268085 - 2.17589188
    r1 = r1 * inverse_square - 5.53605914e-01
    r1 = r1 * inverse_square - 9.87132732e-03
    s1 = inverse_square * -5.77397496e-02 + 1.43113089
    s1 = s1 * inverse_square + 6.69798088
    s1 = s1 * inverse_square + 5.45995426
    s1 = s1 * inverse_square + 1.0
    r2 = inverse_square * -9.53764343 - 1.66696873e+01
    r2 = r2 * inverse_square - 6.16498327
    r2 = r2 * inverse_square - 6.25171244e-01
    r2 = r2 * inverse_square - 9.86494310e-03
    s2 = inverse_square * 8.93033314 + 4.72810211e+01
    s2 = s2 * inverse_square + 4.51839523e+01
    s2 = s2 * inverse_square + 1.26884899e+01
    s2 = s2 * inverse_square + 1.0
    rational = ct.where(x < 2.857142857142857, r1 / s1, r2 / s2)
    high = ct.bitcast(ct.bitcast(x, ct.uint32) & 0xFFFFE000, ct.float32)
    return ct.exp(-high * high - 0.5625) * ct.exp(
        (high - x) * (high + x) + rational
    ) / x


@ct.function
def erf(x):
    # Use the single-precision library algorithm; narrower storage types use
    # the same f32-to-storage conversion as the other ordinary unary functions.
    wide = ct.astype(x, ct.float32)
    magnitude = ct.abs(wide)
    small = _erf_small(ct.minimum(magnitude, 0.84375))
    near = _erf_near_one(ct.minimum(ct.maximum(magnitude, 0.84375), 1.25))
    tail = _erfc_tail(ct.minimum(ct.maximum(magnitude, 1.25), 4.0))
    value = ct.where(magnitude < 0.84375, small,
                     ct.where(magnitude < 1.25, near, 1.0 - tail))
    value = ct.where(magnitude >= 4.0, 1.0, value)
    value = ct.where(wide < 0.0, -value, value)
    value = ct.where((wide == 0.0) | ct.isnan(wide), wide, value)
    return ct.astype(ct.astype(value, ct.float32), x.dtype)


@ct.function
def _polynomial(x, coefficients: ct.Constant):
    value = ct.full(x.shape, coefficients[0], dtype=x.dtype)
    for coefficient in ct.static_iter(coefficients[1:]):
        value = value * x + coefficient
    return value


@ct.function
def erfc(x):
    # Double evaluation leaves ample error margin before the required f32
    # rounding. The positive tail is evaluated directly, without cancellation.
    wide = ct.astype(x, ct.float64)
    magnitude = ct.abs(wide)
    small_x = ct.minimum(magnitude, 0.84375)
    square = small_x * small_x
    numerator = _polynomial(square, (
        -2.37630166566501626084e-05, -5.77027029648944159157e-03,
        -2.84817495755985104766e-02, -3.25042107247001499370e-01,
        1.28379167095512558561e-01,
    ))
    denominator = _polynomial(square, (
        -3.96022827877536812320e-06, 1.32494738004321644526e-04,
        5.08130628187576562776e-03, 6.50222499887672944485e-02,
        3.97917223959155352819e-01, 1.0,
    ))
    small = 0.5 - ((small_x - 0.5) + small_x * (numerator / denominator))
    shifted = ct.minimum(ct.maximum(magnitude, 0.84375), 1.25) - 1.0
    numerator = _polynomial(shifted, (
        -2.16637559486879084300e-03, 3.54783043256182359371e-02,
        -1.10894694282396677476e-01, 3.18346619901161753674e-01,
        -3.72207876035701323847e-01, 4.14856118683748331666e-01,
        -2.36211856075265944077e-03,
    ))
    denominator = _polynomial(shifted, (
        1.19844998467991074170e-02, 1.36370839120290507362e-02,
        1.26171219808761642112e-01, 7.18286544141962662868e-02,
        5.40397917702171048937e-01, 1.06420880400844228286e-01, 1.0,
    ))
    near = (1.0 - 8.45062911510467529297e-01) - numerator / denominator
    tail_x = ct.minimum(ct.maximum(magnitude, 1.25), 28.0)
    inverse_square = 1.0 / (tail_x * tail_x)
    r1 = _polynomial(inverse_square, (
        -9.81432934416914548592e+00, -8.12874355063065934246e+01,
        -1.84605092906711035994e+02, -1.62396669462573470355e+02,
        -6.23753324503260060396e+01, -1.05586262253232909814e+01,
        -6.93858572707181764372e-01, -9.86494403484714822705e-03,
    ))
    s1 = _polynomial(inverse_square, (
        -6.04244152148580987438e-02, 6.57024977031928170135e+00,
        1.08635005541779435134e+02, 4.29008140027567833386e+02,
        6.45387271733267880336e+02, 4.34565877475229228821e+02,
        1.37657754143519042600e+02, 1.96512716674392571292e+01, 1.0,
    ))
    r2 = _polynomial(inverse_square, (
        -4.83519191608651397019e+02, -1.02509513161107724954e+03,
        -6.37566443368389627722e+02, -1.60636384855821916062e+02,
        -1.77579549177547519889e+01, -7.99283237680523006574e-01,
        -9.86494292470009928597e-03,
    ))
    s2 = _polynomial(inverse_square, (
        -2.24409524465858183362e+01, 4.74528541206955367215e+02,
        2.55305040643316442583e+03, 3.19985821950859553908e+03,
        1.53672958608443695994e+03, 3.25792512996573918826e+02,
        3.03380607434824582924e+01, 1.0,
    ))
    rational = ct.where(tail_x < 1.0 / 0.35, r1 / s1, r2 / s2)
    # A finite f32 input has an exactly representable square in f64 here.
    tail = ct.exp(-tail_x * tail_x - 0.5625 + rational) / tail_x
    value = ct.where(magnitude < 0.84375, small,
                     ct.where(magnitude < 1.25, near, tail))
    value = ct.where(magnitude >= 28.0, 0.0, value)
    value = ct.where(wide < 0.0, 2.0 - value, value)
    value = ct.where(ct.isnan(wide), wide, value)
    return ct.astype(ct.astype(value, ct.float32), x.dtype)


@ct.function
def _chebyshev(x, coefficients: ct.Constant):
    current = ct.full(x.shape, coefficients[0], dtype=x.dtype)
    previous = ct.full(x.shape, 0.0, dtype=x.dtype)
    second_previous = previous
    for coefficient in ct.static_iter(coefficients[1:]):
        second_previous = previous
        previous = current
        current = x * previous - second_previous + coefficient
    return 0.5 * (current - second_previous)


@ct.function
def i0(x):
    # Cephes Chebyshev coefficients, also used by ATen's CUDA i0:
    # https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/Math.cuh
    # f64 evaluation avoids exp(f32) overflowing before the f32 I0 result.
    wide = ct.astype(x, ct.float64)
    magnitude = ct.abs(wide)
    small_x = ct.minimum(magnitude, 8.0)
    small = ct.exp(small_x) * _chebyshev(0.5 * small_x - 2.0, (
        -4.41534164647933937950e-18, 3.33079451882223809783e-17,
        -2.43127984654795469359e-16, 1.71539128555513303061e-15,
        -1.16853328779934516808e-14, 7.67618549860493561688e-14,
        -4.85644678311192946090e-13, 2.95505266312963983461e-12,
        -1.72682629144155570723e-11, 9.67580903537323691224e-11,
        -5.18979560163526290666e-10, 2.65982372468238665035e-09,
        -1.30002500998624804212e-08, 6.04699502254191894932e-08,
        -2.67079385394061173391e-07, 1.11738753912010371815e-06,
        -4.41673835845875056359e-06, 1.64484480707288970893e-05,
        -5.75419501008210370398e-05, 1.88502885095841655729e-04,
        -5.76375574538582365885e-04, 1.63947561694133579842e-03,
        -4.32430999505057594430e-03, 1.05464603945949983183e-02,
        -2.37374148058994688156e-02, 4.93052842396707084878e-02,
        -9.49010970480476444210e-02, 1.71620901522208775349e-01,
        -3.04682672343198398683e-01, 6.76795274409476084995e-01,
    ))
    large_x = ct.maximum(magnitude, 8.0)
    large = ct.exp(large_x) * _chebyshev(32.0 / large_x - 2.0, (
        -7.23318048787475395456e-18, -4.83050448594418207126e-18,
        4.46562142029675999901e-17, 3.46122286769746109310e-17,
        -2.82762398051658348494e-16, -3.42548561967721913462e-16,
        1.77256013305652638360e-15, 3.81168066935262242075e-15,
        -9.55484669882830764870e-15, -4.15056934728722208663e-14,
        1.54008621752140982691e-14, 3.85277838274214270114e-13,
        7.18012445138366623367e-13, -1.79417853150680611778e-12,
        -1.32158118404477131188e-11, -3.14991652796324136454e-11,
        1.18891471078464383424e-11, 4.94060238822496958910e-10,
        3.39623202570838634515e-09, 2.26666899049817806459e-08,
        2.04891858946906374183e-07, 2.89137052083475648297e-06,
        6.88975834691682398426e-05, 3.36911647825569408990e-03,
        8.04490411014108831608e-01,
    )) / ct.sqrt(large_x)
    value = ct.where(magnitude <= 8.0, small, large)
    value = ct.where(magnitude == 0.0, 1.0, value)
    value = ct.where(magnitude == float("inf"), float("inf"), value)
    value = ct.where(ct.isnan(wide), wide, value)
    return ct.astype(ct.astype(value, ct.float32), x.dtype)


@ct.function
def log1p(x):
    wide = ct.astype(x, ct.float64)
    shifted = 1.0 + wide
    # When double addition rounds to one, log1p(x) rounds to x in f32.
    value = ct.where(shifted == 1.0, wide, ct.log(shifted))
    return ct.astype(ct.astype(value, ct.float32), x.dtype)


# Lanczos13m53 coefficients from Boost.Math, copyright John Maddock 2006.
# https://github.com/boostorg/math/blob/boost-1.85.0/include/boost/math/special_functions/lanczos.hpp
# Boost Software License - Version 1.0 - August 17th, 2003
# Permission is hereby granted, free of charge, to any person or organization
# obtaining a copy of the software and accompanying documentation covered by
# this license (the "Software") to use, reproduce, display, distribute,
# execute, and transmit the Software, and to prepare derivative works of the
# Software, and to permit third-parties to whom the Software is furnished to
# do so, all subject to the following:
# The copyright notices in the Software and this entire statement, including
# the above license grant, this restriction and the following disclaimer,
# must be included in all copies of the Software, in whole or in part, and
# all derivative works of the Software, unless such copies or derivative
# works are solely in the form of machine-executable object code generated
# by a source language processor.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
# SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
# FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


@ct.function
def lgamma(x):
    wide = ct.astype(x, ct.float64)
    positive = ct.where(wide < 0.5, 1.0 - wide, wide)
    inverse = 1.0 / positive
    # Evaluate the reversed rational function at 1/z to avoid polynomial
    # overflow for the full finite f32 input range.
    numerator = inverse * 56906521.91347156388090791033559122686859 + 103794043.1163445451906271053616070238554
    numerator = numerator * inverse + 86363131.28813859145546927288977868422342
    numerator = numerator * inverse + 43338889.32467613834773723740590533316085
    numerator = numerator * inverse + 14605578.08768506808414169982791359218571
    numerator = numerator * inverse + 3481712.15498064590882071018964774556468
    numerator = numerator * inverse + 601859.6171681098786670226533699352302507
    numerator = numerator * inverse + 75999.29304014542649875303443598909137092
    numerator = numerator * inverse + 6955.999602515376140356310115515198987526
    numerator = numerator * inverse + 449.9445569063168119446858607650988409623
    numerator = numerator * inverse + 19.51992788247617482847860966235652136208
    numerator = numerator * inverse + 0.5098416655656676188125178644804694509993
    numerator = numerator * inverse + 0.006061842346248906525783753964555936883222
    denominator = inverse * 39916800.0 + 120543840.0
    denominator = denominator * inverse + 150917976.0
    denominator = denominator * inverse + 105258076.0
    denominator = denominator * inverse + 45995730.0
    denominator = denominator * inverse + 13339535.0
    denominator = denominator * inverse + 2637558.0
    denominator = denominator * inverse + 357423.0
    denominator = denominator * inverse + 32670.0
    denominator = denominator * inverse + 1925.0
    denominator = denominator * inverse + 66.0
    denominator = denominator * inverse + 1.0
    shifted = positive + 5.524680040776729583740234375
    value = (positive - 0.5) * (ct.log(shifted) - 1.0) + ct.log(numerator / denominator)
    # Reducing around the nearest integer also preserves tiny negative inputs;
    # reducing into [0, 1) would round their fraction to one.
    fraction = wide - ct.floor(wide + 0.5)
    sine = ct.abs(ct.sin(fraction * 3.141592653589793238462643383279502884))
    reflected = 1.144729885849400174143427351353058712 - ct.log(sine) - value
    value = ct.where(wide < 0.5, reflected, value)
    value = ct.where((wide == 1.0) | (wide == 2.0), 0.0, value)
    pole = (wide <= 0.0) & (wide == ct.floor(wide))
    value = ct.where(pole | (ct.abs(wide) == float("inf")), float("inf"), value)
    value = ct.where(ct.isnan(wide), wide, value)
    return ct.astype(ct.astype(value, ct.float32), x.dtype)
