import torch
import intent
import intent.language as I


@intent.kernel
def _linear_relu(
    input: I.In[I.f32, (32, 4096)],
    weight: I.In[I.f32, (2048, 4096)],
    bias: I.In[I.f32, (2048,)],
    activation: I.Out[I.f32, (32, 2048)],
):
    rows = I.domain(0, 32)
    features = I.domain(0, 2048)
    inputs = I.domain(0, 4096)

    x = input[rows, inputs]
    w = weight[features, inputs]
    linear = I.matmul(x, w, acc_dtype=I.f32, transpose_rhs=True)
    shifted = linear + bias[features]
    zeros = I.full(shifted.shape, 0.0, dtype=I.f32)
    activation[rows, features] = I.maximum(shifted, zeros)


@intent.kernel
def _linear_relu_no_bias(
    input: I.In[I.f32, (32, 4096)],
    weight: I.In[I.f32, (2048, 4096)],
    activation: I.Out[I.f32, (32, 2048)],
):
    rows = I.domain(0, 32)
    features = I.domain(0, 2048)
    inputs = I.domain(0, 4096)

    x = input[rows, inputs]
    w = weight[features, inputs]
    linear = I.matmul(x, w, acc_dtype=I.f32, transpose_rhs=True)
    zeros = I.full(linear.shape, 0.0, dtype=I.f32)
    activation[rows, features] = I.maximum(linear, zeros)


@intent.kernel
def _layer_norm(
    activation: I.In[I.f32, (32, 2048)],
    output: I.Out[I.f32, (32, 2048)],
    eps: I.f32,
):
    rows = I.domain(0, 32)
    features = I.domain(0, 2048)
    values = activation[rows, features]

    inv_features = I.cast(2048, I.f32)
    mean = I.fdiv(I.reduce.sum(values, axis=1, acc_dtype=I.f32), inv_features)
    mean_2d = I.reshape(mean, (32, 1))
    centered = values - mean_2d
    variance = I.fdiv(
        I.reduce.sum(centered * centered, axis=1, acc_dtype=I.f32),
        inv_features,
    )
    scale = I.reshape(I.rsqrt(variance + eps), (32, 1))
    output[rows, features] = centered * scale


def build(context):
    linear_relu = context.compile("linear_relu", _linear_relu)
    linear_relu_no_bias = context.compile("linear_relu_no_bias", _linear_relu_no_bias)
    layer_norm = context.compile("layer_norm", _layer_norm)

    def fused_layer_norm_relu_linear(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
        normalized_shape: torch.Size = None,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> torch.Tensor:
        del normalized_shape, elementwise_affine
        activation = torch.empty(
            (32, 2048), device=input.device, dtype=input.dtype
        )
        output = torch.empty(
            (32, 2048), device=input.device, dtype=input.dtype
        )
        if bias is None:
            linear_relu_no_bias(input, weight, activation)
        else:
            linear_relu(input, weight, bias, activation)
        layer_norm(activation, output, eps)
        return output

    return fused_layer_norm_relu_linear
