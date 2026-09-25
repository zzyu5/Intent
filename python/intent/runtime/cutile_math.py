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
def _erf_small_ratio(x):
    square = x * x
    numerator = square * -1.86260219e-03 - 3.36030394e-01
    numerator = numerator * square + 1.28379166e-01
    denominator = square * -1.98859419e-03 + 2.16070302e-02
    denominator = denominator * square + 3.12324286e-01
    denominator = denominator * square + 1.0
    return numerator / denominator


@ct.function
def _erf_small(x):
    value = x + x * _erf_small_ratio(x)
    tiny = (8.0 * x + 1.0270333290 * x) / 8.0
    return ct.where(x < 2.0 ** -119, tiny, value)


@ct.function
def _erf_near_one_ratio(x):
    shifted = x - 1.0
    numerator = shifted * 1.10914491e-01 - 1.65179938e-01
    numerator = numerator * shifted + 4.15109694e-01
    numerator = numerator * shifted + 3.64939137e-06
    denominator = shifted * 5.62181212e-02 + 1.68576106e-01
    denominator = denominator * shifted + 5.35934687e-01
    denominator = denominator * shifted + 6.02074385e-01
    denominator = denominator * shifted + 1.0
    return numerator / denominator


@ct.function
def _erf_near_one(x):
    return 8.42697144e-01 + _erf_near_one_ratio(x)


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
    first_interval = x < 2.857142857142857
    rational = ct.where(first_interval, r1, r2) / ct.where(first_interval, s1, s2)
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
def erfc(x):
    wide = ct.astype(x, ct.float32)
    magnitude = ct.abs(wide)
    small_x = ct.minimum(ct.maximum(wide, -0.84375), 0.84375)
    correction = small_x * _erf_small_ratio(small_x)
    small = ct.where(small_x < 0.25,
                     1.0 - (small_x + correction),
                     0.5 - ((small_x - 0.5) + correction))
    small = ct.where(magnitude < 2.0 ** -24, 1.0 - wide, small)
    near_ratio = _erf_near_one_ratio(
        ct.minimum(ct.maximum(magnitude, 0.84375), 1.25))
    near = ct.where(wide < 0.0,
                    1.0 + (8.42697144e-01 + near_ratio),
                    (1.0 - 8.42697144e-01) - near_ratio)
    # erfc evaluates the positive tail directly. Unlike erf, it must retain
    # subnormal results beyond x=4; fdlibm's f32 tail interval extends to 11.
    tail = _erfc_tail(ct.minimum(ct.maximum(magnitude, 1.25), 11.0))
    tail = ct.where(magnitude >= 11.0, 0.0, tail)
    tail = ct.where(wide < 0.0, 2.0 - tail, tail)
    value = ct.where(magnitude < 0.84375, small,
                     ct.where(magnitude < 1.25, near, tail))
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
    # On the reduced [-2, 2] intervals, omitted leading coefficients bound
    # the series error below 1e-9 (small) and 2e-12 (large), below f32 precision.
    wide = ct.astype(x, ct.float64)
    magnitude = ct.abs(wide)
    small_x = ct.minimum(magnitude, 8.0)
    small = ct.exp(small_x) * _chebyshev(0.5 * small_x - 2.0, (
        2.65982372468238665035e-09,
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
    value = small
    if ct.max(ct.astype(magnitude > 8.0, ct.int32)) != 0:
        large_x = ct.maximum(magnitude, 8.0)
        large = ct.exp(large_x) * _chebyshev(32.0 / large_x - 2.0, (
            -1.79417853150680611778e-12,
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
