import torch
import intent
import intent.language as I


@intent.kernel
def _log_softmax_linear_bias(
    input: I.In[I.f32, (32, 256)],
    weight: I.In[I.f32, (256, 256)],
    bias: I.In[I.f32, (256,)],
    output: I.Out[I.f32, (32, 256)],
):
    scores = I.matmul(input, weight, acc_dtype=I.f32, transpose_rhs=True) + bias

    rows = I.domain(0, 32)
    for row in I.parallel(rows):
        row_scores = scores[row, :]
        row_max = I.reduce.max(row_scores, axis=0, acc_dtype=I.f32)
        shifted = row_scores - row_max
        normalizer = I.reduce.sum(I.exp(shifted), axis=0, acc_dtype=I.f32)
        output[row, :] = shifted - I.log(normalizer)


@intent.kernel
def _log_softmax_linear_no_bias(
    input: I.In[I.f32, (32, 256)],
    weight: I.In[I.f32, (256, 256)],
    output: I.Out[I.f32, (32, 256)],
):
    scores = I.matmul(input, weight, acc_dtype=I.f32, transpose_rhs=True)

    rows = I.domain(0, 32)
    for row in I.parallel(rows):
        row_scores = scores[row, :]
        row_max = I.reduce.max(row_scores, axis=0, acc_dtype=I.f32)
        shifted = row_scores - row_max
        normalizer = I.reduce.sum(I.exp(shifted), axis=0, acc_dtype=I.f32)
        output[row, :] = shifted - I.log(normalizer)


def build(context):
    with_bias = context.compile("log_softmax_linear_bias", _log_softmax_linear_bias)
    without_bias = context.compile("log_softmax_linear_no_bias", _log_softmax_linear_no_bias)

    def log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None):
        if bias is None:
            return without_bias.run(input, weight)
        return with_bias.run(input, weight, bias)

    return log_softmax_linear
