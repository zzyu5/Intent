import intent
import intent.language as I


@intent.kernel
def linear_kernel(
    input: I.In[I.f32, ("M", "K")],
    weight: I.In[I.f32, ("N", "K")],
    output: I.Out[I.f32, ("M", "N")],
):
    rows = I.domain(0, input.shape[0])
    features = I.domain(0, weight.shape[0])
    output[rows, features] = I.matmul(
        input,
        weight,
        transpose_rhs=True,
        acc_dtype=I.f32,
    )


@intent.kernel
def relu_bias_kernel(
    input: I.In[I.f32, ("M", "N")],
    bias: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("M", "N")],
):
    rows = I.domain(0, input.shape[0])
    features = I.domain(0, input.shape[1])
    for row in I.parallel(rows):
        output[row, features] = I.maximum(
            input[row, features] + bias[features],
            0.0,
        )


@intent.kernel
def layer_norm_kernel(
    input: I.In[I.f32, ("M", "N")],
    eps: I.f32,
    output: I.Out[I.f32, ("M", "N")],
):
    rows = I.domain(0, input.shape[0])
    features = I.domain(0, input.shape[1])
    count = I.cast(input.shape[1], I.f32)

    for row in I.parallel(rows):
        values = input[row, features]
        mean = I.fdiv(
            I.reduce.sum(values, axis=0, acc_dtype=I.f32),
            count,
        )
        centered = values - mean
        variance = I.fdiv(
            I.reduce.sum(centered * centered, axis=0, acc_dtype=I.f32),
            count,
        )
        output[row, features] = I.fdiv(
            centered,
            (variance + eps) ** 0.5,
        )


def build(context):
    linear = context.compile("fused_linear_generation", linear_kernel)
    relu_bias = context.compile("fused_relu_bias_generation", relu_bias_kernel)
    layer_norm = context.compile("fused_layer_norm_generation", layer_norm_kernel)

    def wrapper(input, weight, bias=None, normalized_shape=None, eps=1e-5, elementwise_affine=True):
        linear_output = linear.run(input, weight)
        activated = relu_bias.run(linear_output, bias)
        return layer_norm.run(activated, eps)

    return wrapper
