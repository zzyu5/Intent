from __future__ import annotations

import torch
import intent
import intent.language as I


@intent.kernel
def _gelu_none(
    input: I.In[I.f32, (1048576,)],
    mask: I.In[I.bool, (1048576,)],
    output: I.Out[I.f32, (524236,)],
    other: I.f32,
    alpha: I.f32,
):
    # The inclusive scan supplies a stable, collision-free destination rank.
    prefix = I.cumsum(mask, axis=0, acc_dtype=I.i64)
    for source in I.parallel(I.domain(0, 1048576)):
        selected = mask[source]
        if selected:
            destination = I.cast(prefix[source] - 1, I.index)
            value = input[source] + alpha * other
            output[destination] = 0.5 * value * (
                1.0 + I.erf(value * 0.7071067811865476)
            )


@intent.kernel
def _gelu_tanh(
    input: I.In[I.f32, (1048576,)],
    mask: I.In[I.bool, (1048576,)],
    output: I.Out[I.f32, (524236,)],
    other: I.f32,
    alpha: I.f32,
):
    prefix = I.cumsum(mask, axis=0, acc_dtype=I.i64)
    for source in I.parallel(I.domain(0, 1048576)):
        selected = mask[source]
        if selected:
            destination = I.cast(prefix[source] - 1, I.index)
            value = input[source] + alpha * other
            cubic = value * value * value
            argument = 0.7978845608028654 * (value + 0.044715 * cubic)
            output[destination] = 0.5 * value * (
                1.0 + I.tanh(argument, approximate=True)
            )


def build(context):
    exact = context.compile("masked_select_add_gelu_none", _gelu_none)
    tanh = context.compile("masked_select_add_gelu_tanh", _gelu_tanh)

    def fused_masked_select_add_gelu(
        input,
        mask,
        other,
        *,
        alpha=1,
        approximate="none",
        out=None,
    ):
        if out is None:
            out = torch.empty((524236,), device=input.device, dtype=input.dtype)

        artifact = exact if approximate == "none" else tanh
        artifact(input, mask, out, float(other), float(alpha))
        return out

    return fused_masked_select_add_gelu
