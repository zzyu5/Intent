import torch
import intent
import intent.language as I


@intent.kernel
def _silu_batch_norm_affine(
    input: I.In[I.f32, (16, 256, 8, 128)],
    running_mean: I.In[I.f32, (256,)],
    running_var: I.In[I.f32, (256,)],
    weight: I.In[I.f32, (256,)],
    bias: I.In[I.f32, (256,)],
    output: I.Out[I.f32, (16, 256, 8, 128)],
    eps: I.f32,
):
    n = I.domain(0, 16)
    c = I.domain(0, 256)
    h = I.domain(0, 8)
    w = I.domain(0, 128)

    x = input[n, c, h, w]
    mean = I.reshape(running_mean[c], (1, 256, 1, 1))
    inv_std = I.reshape(I.rsqrt(running_var[c] + eps), (1, 256, 1, 1))
    scale = I.reshape(weight[c], (1, 256, 1, 1))
    shift = I.reshape(bias[c], (1, 256, 1, 1))

    normalized = (x - mean) * inv_std
    activated_input = normalized * scale + shift
    output[n, c, h, w] = activated_input * I.sigmoid(activated_input)


@intent.kernel
def _silu_batch_norm_weight_only(
    input: I.In[I.f32, (16, 256, 8, 128)],
    running_mean: I.In[I.f32, (256,)],
    running_var: I.In[I.f32, (256,)],
    weight: I.In[I.f32, (256,)],
    output: I.Out[I.f32, (16, 256, 8, 128)],
    eps: I.f32,
):
    n = I.domain(0, 16)
    c = I.domain(0, 256)
    h = I.domain(0, 8)
    w = I.domain(0, 128)

    x = input[n, c, h, w]
    mean = I.reshape(running_mean[c], (1, 256, 1, 1))
    inv_std = I.reshape(I.rsqrt(running_var[c] + eps), (1, 256, 1, 1))
    scale = I.reshape(weight[c], (1, 256, 1, 1))

    normalized = (x - mean) * inv_std
    activated_input = normalized * scale
    output[n, c, h, w] = activated_input * I.sigmoid(activated_input)


@intent.kernel
def _silu_batch_norm_bias_only(
    input: I.In[I.f32, (16, 256, 8, 128)],
    running_mean: I.In[I.f32, (256,)],
    running_var: I.In[I.f32, (256,)],
    bias: I.In[I.f32, (256,)],
    output: I.Out[I.f32, (16, 256, 8, 128)],
    eps: I.f32,
):
    n = I.domain(0, 16)
    c = I.domain(0, 256)
    h = I.domain(0, 8)
    w = I.domain(0, 128)

    x = input[n, c, h, w]
    mean = I.reshape(running_mean[c], (1, 256, 1, 1))
    inv_std = I.reshape(I.rsqrt(running_var[c] + eps), (1, 256, 1, 1))
    shift = I.reshape(bias[c], (1, 256, 1, 1))

    normalized = (x - mean) * inv_std
    activated_input = normalized + shift
    output[n, c, h, w] = activated_input * I.sigmoid(activated_input)


@intent.kernel
def _silu_batch_norm_no_affine(
    input: I.In[I.f32, (16, 256, 8, 128)],
    running_mean: I.In[I.f32, (256,)],
    running_var: I.In[I.f32, (256,)],
    output: I.Out[I.f32, (16, 256, 8, 128)],
    eps: I.f32,
):
    n = I.domain(0, 16)
    c = I.domain(0, 256)
    h = I.domain(0, 8)
    w = I.domain(0, 128)

    x = input[n, c, h, w]
    mean = I.reshape(running_mean[c], (1, 256, 1, 1))
    inv_std = I.reshape(I.rsqrt(running_var[c] + eps), (1, 256, 1, 1))

    normalized = (x - mean) * inv_std
    output[n, c, h, w] = normalized * I.sigmoid(normalized)


def build(context):
    affine_kernel = context.compile("silu_batch_norm_affine", _silu_batch_norm_affine)
    weight_only_kernel = context.compile(
        "silu_batch_norm_weight_only", _silu_batch_norm_weight_only
    )
    bias_only_kernel = context.compile(
        "silu_batch_norm_bias_only", _silu_batch_norm_bias_only
    )
    no_affine_kernel = context.compile("silu_batch_norm_no_affine", _silu_batch_norm_no_affine)

    def silu_batch_norm(
        input,
        running_mean,
        running_var,
        weight=None,
        bias=None,
        training=False,
        momentum=0.1,
        eps=1e-5,
    ):
        if training:
            raise NotImplementedError("the fixed profile uses inference-mode BatchNorm")
        del momentum

        if weight is None:
            if bias is None:
                return no_affine_kernel.run(input, running_mean, running_var, eps)
            return bias_only_kernel.run(input, running_mean, running_var, bias, eps)
        if bias is None:
            return weight_only_kernel.run(input, running_mean, running_var, weight, eps)
        return affine_kernel.run(input, running_mean, running_var, weight, bias, eps)

    return silu_batch_norm
