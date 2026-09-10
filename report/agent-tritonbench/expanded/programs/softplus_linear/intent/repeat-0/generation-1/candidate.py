import math

import intent
import intent.language as I


@intent.fn
def _softplus(value, beta, threshold):
    scaled = value * beta
    exact = math.log1p(I.exp2(scaled * 1.4426950408889634)) / beta
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
