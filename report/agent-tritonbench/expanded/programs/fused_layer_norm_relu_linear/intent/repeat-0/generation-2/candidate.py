import intent
import intent.language as I


@intent.kernel
def fused_layer_norm_relu_linear_kernel(
    input: I.In[I.f32, ("M", "K")],
    weight: I.In[I.f32, ("N", "K")],
    bias: I.In[I.f32, ("N",)],
    eps: I.f32,
    output: I.Out[I.f32, ("M", "N")],
):
    rows = I.domain(0, input.shape[0])
    features = I.domain(0, weight.shape[0])

    linear = I.matmul(
        input,
        weight,
        transpose_rhs=True,
        acc_dtype=I.f32,
    )
    count = I.cast(weight.shape[0], I.f32)
    activated = I.maximum(linear + bias, 0.0)

    mean = I.fdiv(
        I.reduce.sum(activated, axis=1, acc_dtype=I.f32),
        count,
    )
    mean = I.reshape(mean, (input.shape[0], 1))
    centered = activated - mean

    variance = I.fdiv(
        I.reduce.sum(centered * centered, axis=1, acc_dtype=I.f32),
        count,
    )
    variance = I.reshape(variance, (input.shape[0], 1))
    output[rows, features] = centered / (variance + eps) ** 0.5


def build(context):
    compiled = context.compile(
        "fused_layer_norm_relu_linear_generation",
        fused_layer_norm_relu_linear_kernel,
    )

    def wrapper(input, weight, bias=None, normalized_shape=None, eps=1e-5, elementwise_affine=True):
        return compiled.run(input, weight, bias, eps)

    return wrapper
