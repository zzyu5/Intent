import intent
import intent.language as I


@intent.fn
def _softplus(value, beta, threshold):
    scaled = value * beta

    # log(1 + exp(x)) = max(x, 0) + log1p(exp(-abs(x))).
    # Express the max/abs with select so the helper only uses scalar
    # arithmetic and the documented exp2 operation.
    positive = scaled if scaled > 0.0 else scaled - scaled
    magnitude = scaled if scaled > 0.0 else -scaled
    exp_neg = I.exp2((-magnitude) * 1.4426950408889634)
    t = exp_neg / (2.0 + exp_neg)
    t2 = t * t

    # log1p(z) = 2 * atanh(z / (2 + z)); z <= 1, so this short
    # odd-power expansion is accurate in float32 across the full range.
    series = 0.043478260869565216
    series = 0.047619047619047616 + t2 * series
    series = 0.05263157894736842 + t2 * series
    series = 0.058823529411764705 + t2 * series
    series = 0.06666666666666667 + t2 * series
    series = 0.07692307692307693 + t2 * series
    series = 0.09090909090909091 + t2 * series
    series = 0.1111111111111111 + t2 * series
    series = 0.14285714285714285 + t2 * series
    series = 0.2 + t2 * series
    series = 0.3333333333333333 + t2 * series
    exact = (positive + 2.0 * t * (1.0 + t2 * series)) / beta
    return value if scaled > threshold else exact


@intent.kernel
def _softplus_linear_bias(
    input: I.In[I.f32, ("M", "K")],
    weight: I.In[I.f32, ("N", "K")],
    bias: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("M", "N")],
    beta: I.f32,
    threshold: I.f32,
):
    linear = I.matmul(input, weight, acc_dtype=I.f32, transpose_rhs=True)
    rows = I.domain(0, input.shape[0])
    columns = I.domain(0, weight.shape[0])
    output[rows, columns] = _softplus(linear + bias, beta, threshold)


@intent.kernel
def _softplus_linear_no_bias(
    input: I.In[I.f32, ("M", "K")],
    weight: I.In[I.f32, ("N", "K")],
    output: I.Out[I.f32, ("M", "N")],
    beta: I.f32,
    threshold: I.f32,
):
    linear = I.matmul(input, weight, acc_dtype=I.f32, transpose_rhs=True)
    rows = I.domain(0, input.shape[0])
    columns = I.domain(0, weight.shape[0])
    output[rows, columns] = _softplus(linear, beta, threshold)


def build(context):
    with_bias = context.compile("softplus_linear_bias", _softplus_linear_bias)
    without_bias = context.compile("softplus_linear_no_bias", _softplus_linear_no_bias)

    def wrapper(input, weight, bias=None, beta=1, threshold=20):
        beta = float(beta)
        threshold = float(threshold)
        if bias is None:
            return without_bias.run(input, weight, beta, threshold)
        return with_bias.run(input, weight, bias, beta, threshold)

    return wrapper
