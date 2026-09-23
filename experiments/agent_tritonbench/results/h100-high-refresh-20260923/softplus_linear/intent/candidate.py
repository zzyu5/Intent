import torch
import intent
import intent.language as I


@intent.fn
def _thresholded_softplus(value, beta: I.f32, threshold: I.f32):
    # The threshold is deliberately applied to the unscaled linear result.
    softplus = I.log1p(I.exp(beta * value)) / beta
    return I.select(value > threshold, value, softplus)


@intent.kernel
def _softplus_linear_bias(
    input: I.In[I.f32, ("M", "K")],
    weight: I.In[I.f32, ("N", "K")],
    bias: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("M", "N")],
    beta: I.f32,
    threshold: I.f32,
):
    M, _ = input.shape
    N, _ = weight.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)

    linear = I.matmul(input, weight, acc_dtype=I.f32, transpose_rhs=True)
    values = _thresholded_softplus(linear + bias, beta, threshold)
    output[rows, columns] = values


@intent.kernel
def _softplus_linear_no_bias(
    input: I.In[I.f32, ("M", "K")],
    weight: I.In[I.f32, ("N", "K")],
    output: I.Out[I.f32, ("M", "N")],
    beta: I.f32,
    threshold: I.f32,
):
    M, _ = input.shape
    N, _ = weight.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)

    linear = I.matmul(input, weight, acc_dtype=I.f32, transpose_rhs=True)
    values = _thresholded_softplus(linear, beta, threshold)
    output[rows, columns] = values


def build(context):
    with_bias = context.compile("softplus_linear_bias", _softplus_linear_bias)
    without_bias = context.compile("softplus_linear_no_bias", _softplus_linear_no_bias)

    def softplus_linear(input, weight, bias=None, beta=1, threshold=20):
        output = torch.empty(
            (input.shape[0], weight.shape[0]),
            device=input.device,
            dtype=input.dtype,
        )
        beta_value = float(beta)
        threshold_value = float(threshold)
        if bias is None:
            without_bias(input, weight, output, beta_value, threshold_value)
        else:
            with_bias(input, weight, bias, output, beta_value, threshold_value)
        return output

    return softplus_linear
