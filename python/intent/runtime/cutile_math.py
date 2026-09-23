"""Tile implementations of ordinary floating-point library operations."""

# The erf rational coefficients and interval decomposition are from fdlibm:
# https://github.com/JuliaMath/openlibm/blob/master/src/s_erff.c
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
