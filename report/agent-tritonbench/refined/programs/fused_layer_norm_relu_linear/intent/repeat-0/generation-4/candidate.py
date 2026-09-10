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
def row_mean(
    input: I.In[I.f32, ("M", "N")],
    output: I.Out[I.f32, ("M",)],
):
    rows = I.domain(0, input.shape[0])
    count = I.cast(input.shape[1], I.f32)
    mean = I.reduce.sum(input, axis=1, acc_dtype=I.f32) / count
    output[rows] = mean


@intent.kernel
def row_variance(
    input: I.In[I.f32, ("M", "N")],
    mean: I.In[I.f32, ("M",)],
    output: I.Out[I.f32, ("M",)],
):
    rows = I.domain(0, input.shape[0])
    mean_matrix = I.reshape(mean, (input.shape[0], 1))
    centered = input - mean_matrix
    count = I.cast(input.shape[1], I.f32)
    variance = I.reduce.sum(centered * centered, axis=1, acc_dtype=I.f32) / count
    output[rows] = variance


@intent.kernel
def normalize_rows(
    input: I.In[I.f32, ("M", "N")],
    mean: I.In[I.f32, ("M",)],
    variance: I.In[I.f32, ("M",)],
    output: I.Out[I.f32, ("M", "N")],
    eps: I.f32,
):
    rows = I.domain(0, input.shape[0])
    columns = I.domain(0, input.shape[1])
    mean_matrix = I.reshape(mean, (input.shape[0], 1))
    variance_matrix = I.reshape(variance, (input.shape[0], 1))
    normalized = (input - mean_matrix) / I.sqrt(variance_matrix + eps)
    output[rows, columns] = normalized


def build(context):
    linear_bias_compiled = context.compile("fused_linear_relu_bias", linear_relu_bias)
    linear_no_bias_compiled = context.compile("fused_linear_relu_no_bias", linear_relu_no_bias)
    mean_compiled = context.compile("fused_row_mean", row_mean)
    variance_compiled = context.compile("fused_row_variance", row_variance)
    normalize_compiled = context.compile("fused_normalize_rows", normalize_rows)

    def wrapper(input, weight, bias=None, normalized_shape=None, eps=1e-5, elementwise_affine=True):
        if bias is None:
            activated = linear_no_bias_compiled.run(input, weight)
        else:
            activated = linear_bias_compiled.run(input, weight, bias)
        mean = mean_compiled.run(activated)
        variance = variance_compiled.run(activated, mean)
        return normalize_compiled.run(activated, mean, variance, eps)

    return wrapper
