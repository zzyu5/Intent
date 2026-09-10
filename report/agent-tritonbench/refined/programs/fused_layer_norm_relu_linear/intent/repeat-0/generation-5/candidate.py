import intent
import intent.language as I


@intent.kernel
def linear_relu_bias(
    input: I.In[I.f32, ("M", "K")],
    weight: I.In[I.f32, ("N", "K")],
    bias: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("M", "N")],
):
    rows = I.domain(0, input.shape[0])
    columns = I.domain(0, weight.shape[0])
    linear = I.matmul(
        input,
        weight,
        transpose_rhs=True,
        acc_dtype=I.f32,
    )
    activated = I.maximum(linear + bias, I.cast(0.0, I.f32))
    output[rows, columns] = activated


@intent.kernel
def linear_relu_no_bias(
    input: I.In[I.f32, ("M", "K")],
    weight: I.In[I.f32, ("N", "K")],
    output: I.Out[I.f32, ("M", "N")],
):
    rows = I.domain(0, input.shape[0])
    columns = I.domain(0, weight.shape[0])
    linear = I.matmul(
        input,
        weight,
        transpose_rhs=True,
        acc_dtype=I.f32,
    )
    activated = I.maximum(linear, I.cast(0.0, I.f32))
    output[rows, columns] = activated


@intent.kernel
def layer_norm_rows(
    input: I.In[I.f32, ("M", "N")],
    output: I.Out[I.f32, ("M", "N")],
    eps: I.f32,
):
    rows = I.domain(0, input.shape[0])
    columns = I.domain(0, input.shape[1])
    count = I.cast(input.shape[1], I.f32)

    for row in I.parallel(rows):
        values = input[row, columns]
        mean = I.reduce.sum(values, axis=0, acc_dtype=I.f32) / count
        centered = values - mean
        variance = I.reduce.sum(centered * centered, axis=0, acc_dtype=I.f32) / count
        normalized = centered / I.sqrt(variance + eps)
        output[row, columns] = normalized


def build(context):
    linear_bias_compiled = context.compile("fused_linear_relu_bias", linear_relu_bias)
    linear_no_bias_compiled = context.compile("fused_linear_relu_no_bias", linear_relu_no_bias)
    norm_compiled = context.compile("fused_layer_norm_rows", layer_norm_rows)

    def wrapper(input, weight, bias=None, normalized_shape=None, eps=1e-5, elementwise_affine=True):
        if bias is None:
            activated = linear_no_bias_compiled.run(input, weight)
        else:
            activated = linear_bias_compiled.run(input, weight, bias)
        return norm_compiled.run(activated, eps)

    return wrapper
