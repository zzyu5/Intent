import intent
import intent.language as I


@intent.fn
def natural_log(value):
    normalized = value
    exponent = I.cast(0.0, I.f32)
    for _ in range(20):
        if normalized >= 2.0:
            normalized = normalized * 0.5
            exponent = exponent + 1.0

    z = (normalized - 1.0) / (normalized + 1.0)
    z2 = z * z
    series = z
    power = z * z2
    series = series + power / 3.0
    power = power * z2
    series = series + power / 5.0
    power = power * z2
    series = series + power / 7.0
    power = power * z2
    series = series + power / 9.0
    power = power * z2
    series = series + power / 11.0
    power = power * z2
    series = series + power / 13.0
    power = power * z2
    series = series + power / 15.0
    power = power * z2
    series = series + power / 17.0
    return exponent * 0.6931471805599453 + 2.0 * series


@intent.kernel
def logsumexp_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ()],
):
    elements = I.domain(0, input.shape[0])
    maximum = I.reduce.max(input[elements], axis=0)
    shifted = input[elements] - maximum
    total = I.reduce.sum(
        I.exp2(shifted * 1.4426950408889634),
        axis=0,
        acc_dtype=I.f32,
    )
    output[()] = maximum + natural_log(total)


def build(context):
    compiled = context.compile("logsumexp", logsumexp_kernel)

    def wrapper(input, dim, keepdim=False, *, out=None):
        if out is None:
            return compiled.run(input)
        compiled(input, out)
        return out

    return wrapper
