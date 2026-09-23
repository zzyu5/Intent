from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def _add_gelu_tensor(
    input: I.In[I.f16, ("N",)],
    other: I.In[I.f16, ("N",)],
    alpha: I.f32,
    output: I.Out[I.f16, ("N",)],
    APPROX: I.Constexpr[bool],
):
    n = input.shape[0]
    elements = I.domain(0, n)
    x = I.cast(input[elements], I.f32) + I.cast(other[elements], I.f32) * alpha

    if APPROX:
        c = I.cast(0.044715, I.f32)
        half = I.cast(0.5, I.f32)
        one = I.cast(1.0, I.f32)
        cubic = x * x * x
        activated = half * x * (one + I.tanh(x + c * cubic, approximate=True))
    else:
        half = I.cast(0.5, I.f32)
        one = I.cast(1.0, I.f32)
        inv_sqrt_two = I.cast(0.7071067811865475, I.f32)
        activated = half * x * (one + I.erf(x * inv_sqrt_two))

    output[elements] = I.cast(activated, I.f16)


@intent.kernel
def _add_gelu_scalar(
    input: I.In[I.f16, ("N",)],
    other: I.f32,
    alpha: I.f32,
    output: I.Out[I.f16, ("N",)],
    APPROX: I.Constexpr[bool],
):
    n = input.shape[0]
    elements = I.domain(0, n)
    x = I.cast(input[elements], I.f32) + alpha * other

    if APPROX:
        c = I.cast(0.044715, I.f32)
        half = I.cast(0.5, I.f32)
        one = I.cast(1.0, I.f32)
        cubic = x * x * x
        activated = half * x * (one + I.tanh(x + c * cubic, approximate=True))
    else:
        half = I.cast(0.5, I.f32)
        one = I.cast(1.0, I.f32)
        inv_sqrt_two = I.cast(0.7071067811865475, I.f32)
        activated = half * x * (one + I.erf(x * inv_sqrt_two))

    output[elements] = I.cast(activated, I.f16)


def build(context):
    tensor_none = context.compile(
        "add_gelu_tensor_none", _add_gelu_tensor, constexprs={"APPROX": False}
    )
    tensor_tanh = context.compile(
        "add_gelu_tensor_tanh", _add_gelu_tensor, constexprs={"APPROX": True}
    )
    scalar_none = context.compile(
        "add_gelu_scalar_none", _add_gelu_scalar, constexprs={"APPROX": False}
    )
    scalar_tanh = context.compile(
        "add_gelu_scalar_tanh", _add_gelu_scalar, constexprs={"APPROX": True}
    )

    def add_gelu(input, other, alpha=1, approximate="none", out=None):
        if approximate == "none":
            tensor_kernel = tensor_none
            scalar_kernel = scalar_none
        elif approximate == "tanh":
            tensor_kernel = tensor_tanh
            scalar_kernel = scalar_tanh
        else:
            raise ValueError("approximate must be 'none' or 'tanh'")

        result = input if out is None else out
        if out is None:
            result = torch.empty_like(input)

        alpha_value = float(alpha)
        if isinstance(other, torch.Tensor):
            tensor_kernel(input, other, alpha_value, result)
        else:
            scalar_kernel(input, float(other), alpha_value, result)
        return result

    return add_gelu
