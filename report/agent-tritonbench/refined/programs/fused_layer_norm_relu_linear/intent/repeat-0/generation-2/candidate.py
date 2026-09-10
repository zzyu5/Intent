import intent
import intent.language as I


@intent.kernel
def linear_relu(
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
def layer_norm(
    input: I.In[I.f32, ("M", "N")],
    output: I.Out[I.f32, ("M", "N")],
    eps: I.f32,
):
    rows = I.domain(0, input.shape[0])
    columns = I.domain(0, input.shape[1])
    count = I.cast(input.shape[1], I.f32)
    mean = I.reduce.sum(input, axis=1, acc_dtype=I.f32) / count
    mean = I.reshape(mean, (input.shape[0], 1))
    centered = input - mean
    variance = I.reduce.sum(centered * centered, axis=1, acc_dtype=I.f32) / count
    variance = I.reshape(variance, (input.shape[0], 1))
    normalized = centered / I.sqrt(variance + eps)
    output[rows, columns] = normalized


def build(context):
    linear_compiled = context.compile("fused_linear_relu", linear_relu)
    norm_compiled = context.compile("fused_layer_norm", layer_norm)

    def wrapper(input, weight, bias=None, normalized_shape=None, eps=1e-5, elementwise_affine=True):
        linear_output = linear_compiled.run(input, weight, bias)
        return norm_compiled.run(linear_output, eps)

    return wrapper
