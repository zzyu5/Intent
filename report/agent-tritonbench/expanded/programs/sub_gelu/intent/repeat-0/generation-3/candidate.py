import intent
import intent.language as I


@intent.kernel
def sub_gelu_exact(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    x = input[elements] - other[elements]

    # Abramowitz-Stegun's erf approximation is accurate enough for the
    # exact GELU tolerance while using only documented pointwise operations.
    z = x * 0.7071067811865475
    abs_z = I.maximum(z, -z)
    t = 1.0 / (1.0 + 0.3275911 * abs_z)
    polynomial = (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
                    - 0.284496736) * t + 0.254829592) * t)
    exp_term = I.exp2(-abs_z * abs_z * 1.4426950408889634)
    erf_positive = 1.0 - polynomial * exp_term
    # Recover the odd sign without converting a tensor predicate.  The small
    # floor only affects values indistinguishable from zero at the requested
    # output tolerance and keeps erf(0) finite.
    safe_abs_z = I.maximum(abs_z, 1.0e-20)
    erf = I.fdiv(z * erf_positive, safe_abs_z)
    output[elements] = 0.5 * x * (1.0 + erf)


def build(context):
    compiled = context.compile("sub_gelu_exact", sub_gelu_exact)

    def wrapper(input, other, alpha=1, approximate="none", out=None):
        if alpha != 1:
            raise ValueError("the supplied profile requires alpha=1")
        if approximate != "none":
            raise ValueError("the supplied profile requires approximate='none'")
        if out is None:
            return compiled.run(input, other)
        compiled(input, other, out)
        return out

    return wrapper
