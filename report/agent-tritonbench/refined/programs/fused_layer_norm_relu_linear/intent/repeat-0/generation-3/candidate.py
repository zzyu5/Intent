import intent
import intent.language as I


@intent.kernel
def fused_layer_norm_relu_linear(
    input: I.In[I.f32, ("M", "K")],
    weight: I.In[I.f32, ("N", "K")],
    bias: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("M", "N")],
    eps: I.f32,
):
    rows = I.domain(0, input.shape[0])
    columns = I.domain(0, weight.shape[0])

    linear = I.matmul(
        input,
        weight,
        transpose_rhs=True,
        acc_dtype=I.f32,
    )
    bias_matrix = I.reshape(bias, (1, weight.shape[0]))
    activated = I.maximum(
        linear + bias_matrix,
        I.cast(0.0, I.f32),
    )

    count = I.cast(activated.shape[1], I.f32)
    mean = I.reduce.sum(activated, axis=1, acc_dtype=I.f32) / count
    mean = I.reshape(mean, (input.shape[0], 1))
    centered = activated - mean
    variance = I.reduce.sum(centered * centered, axis=1, acc_dtype=I.f32) / count
    variance = I.reshape(variance, (input.shape[0], 1))
    normalized = centered / I.sqrt(variance + eps)

    output[rows, columns] = normalized


def build(context):
    compiled = context.compile(
        "fused_layer_norm_relu_linear",
        fused_layer_norm_relu_linear,
    )

    def wrapper(input, weight, bias=None, normalized_shape=None, eps=1e-5, elementwise_affine=True):
        return compiled.run(input, weight, bias, eps)

    return wrapper
