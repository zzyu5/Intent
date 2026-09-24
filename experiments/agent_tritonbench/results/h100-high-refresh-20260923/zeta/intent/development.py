import torch
import intent
import intent.language as I


@intent.fn
def _positive_zeta(s: I.f32, q: I.f32):
    # Eight direct terms keep the asymptotic point well away from q=0.
    acc = I.cast(0.0, I.f32)
    for n in range(8):
        nf = I.cast(n, I.f32)
        base = q + nf
        acc = acc + I.exp(-s * I.log(base))

    one = I.cast(1.0, I.f32)
    half = I.cast(0.5, I.f32)
    b = q + I.cast(8, I.f32)
    blog = I.log(b)
    power = I.exp(-s * blog)
    acc = acc + I.exp((one - s) * blog) / (s - one)
    acc = acc + half * power

    # Euler-Maclaurin corrections through B_12 / 12!. For large x the
    # direct terms already dominate and skipping these avoids inf * 0.
    if s < I.cast(32.0, I.f32):
        inverse = one / b
        inverse_square = inverse * inverse
        term = s * power * inverse
        acc = acc + I.cast(1.0 / 12.0, I.f32) * term
        term = term * (s + one) * (s + I.cast(2.0, I.f32)) * inverse_square
        acc = acc - I.cast(1.0 / 720.0, I.f32) * term
        term = term * (s + I.cast(3.0, I.f32)) * (s + I.cast(4.0, I.f32)) * inverse_square
        acc = acc + I.cast(1.0 / 30240.0, I.f32) * term
        term = term * (s + I.cast(5.0, I.f32)) * (s + I.cast(6.0, I.f32)) * inverse_square
        acc = acc - I.cast(1.0 / 1209600.0, I.f32) * term
        term = term * (s + I.cast(7.0, I.f32)) * (s + I.cast(8.0, I.f32)) * inverse_square
        acc = acc + I.cast(1.0 / 47900160.0, I.f32) * term
        term = term * (s + I.cast(9.0, I.f32)) * (s + I.cast(10.0, I.f32)) * inverse_square
        acc = acc - I.cast(691.0 / 1307674368000.0, I.f32) * term
    return acc


@intent.kernel
def _zeta_kernel(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
):
    nan = I.fdiv(I.cast(0.0, I.f32), I.cast(0.0, I.f32))
    inf = I.cast(I.inf, I.f32)
    one = I.cast(1.0, I.f32)
    zero = I.cast(0.0, I.f32)

    for i in I.parallel(I.domain(0, input.shape[0])):
        s = input[i]
        q = other[i]
        result = nan

        if s == one:
            result = inf
        elif s < one:
            result = nan
        elif q > zero:
            result = _positive_zeta(s, q)
        else:
            q_is_integer = q == I.floor(q)
            if q_is_integer:
                result = inf
            else:
                s_is_integer = s == I.floor(s)
                if s_is_integer:
                    # Shift q into (0, 1] and add the finite recurrence terms.
                    shift = I.cast(I.floor(-q) + one, I.index)
                    result = _positive_zeta(s, q + I.cast(shift, I.f32))
                    j = I.cast(0, I.index)
                    odd = s - I.floor(s / I.cast(2.0, I.f32)) * I.cast(2.0, I.f32) == one
                    sign = I.cast(I.select(odd, -1.0, 1.0), I.f32)
                    while j < shift:
                        base = q + I.cast(j, I.f32)
                        result = result + sign * I.exp(-s * I.log(I.abs(base)))
                        j = j + I.cast(1, I.index)
                else:
                    result = nan

        out[i] = result


def build(context):
    kernel = context.compile("hurwitz_zeta", _zeta_kernel)

    def zeta(input, other, out=None):
        if out is None:
            out = torch.empty_like(input)
        kernel(input, other, out)
        return out

    return zeta
