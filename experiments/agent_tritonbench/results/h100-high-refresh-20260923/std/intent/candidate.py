import torch
import intent
import intent.language as I


GROUPS = 256
CHUNK = 4096
NUMEL = GROUPS * CHUNK


@intent.kernel
def _partial_sum(
    x: I.In[I.f32, (GROUPS, CHUNK)],
    partial: I.Out[I.f32, (GROUPS,)],
):
    rows = I.domain(0, GROUPS)
    cols = I.domain(0, CHUNK)
    values = x[rows, cols]
    partial[rows] = I.reduce.sum(values, axis=1, acc_dtype=I.f32)


@intent.kernel
def _finish_std(
    x: I.In[I.f32, (GROUPS, CHUNK)],
    partial: I.In[I.f32, (GROUPS,)],
    out: I.Out[I.f32, ()],
    correction: I.i64,
):
    rows = I.domain(0, GROUPS)
    cols = I.domain(0, CHUNK)
    groups = I.domain(0, GROUPS)

    total = I.reduce.sum(partial[groups], axis=0, acc_dtype=I.f32)
    mean = total / I.cast(NUMEL, I.f32)

    values = x[rows, cols]
    centered = values - mean
    squared_sum = I.reduce.sum(centered * centered, axis=(0, 1), acc_dtype=I.f32)

    denominator = I.maximum(
        I.cast(0.0, I.f32),
        I.cast(NUMEL, I.f32) - I.cast(correction, I.f32),
    )
    out[()] = I.sqrt(squared_sum / denominator)


def build(context):
    partial_sum = context.compile("std_partial_sum", _partial_sum)
    finish_std = context.compile("std_finish", _finish_std)

    def std(input, dim=None, *, correction=1, keepdim=False, out=None):
        input_2d = input.view(GROUPS, CHUNK)
        partial = torch.empty((GROUPS,), device=input.device, dtype=torch.float32)
        result = out
        if result is None:
            result = torch.empty((), device=input.device, dtype=torch.float32)

        partial_sum(input_2d, partial)
        finish_std(input_2d, partial, result, correction)
        return result

    return std
