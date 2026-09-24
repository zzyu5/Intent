import torch
import intent
import intent.language as I


@intent.kernel
def _std_group_stats(
    input: I.In[I.f32, (1048576,)],
    group_means: I.Out[I.f32, (256,)],
    group_m2: I.Out[I.f32, (256,)],
):
    groups = I.domain(0, 256)
    for group in I.parallel(groups):
        members = I.domain(group * 4096, (group + 1) * 4096)
        values = input[members]
        total = I.reduce.sum(values, axis=0, acc_dtype=I.f32)
        mean = total / I.cast(4096, I.f32)
        centered = values - mean
        m2 = I.reduce.sum(centered * centered, axis=0, acc_dtype=I.f32)
        group_means[group] = mean
        group_m2[group] = m2


@intent.kernel
def _std_merge(
    group_means: I.In[I.f32, (256,)],
    group_m2: I.In[I.f32, (256,)],
    output: I.Out[I.f32, ()],
    correction: I.i64,
):
    mean_sum = I.reduce.sum(group_means, axis=0, acc_dtype=I.f32)
    mean = mean_sum / I.cast(256, I.f32)
    centered_means = group_means - mean
    between = I.reduce.sum(centered_means * centered_means, axis=0, acc_dtype=I.f32)
    between = between * I.cast(4096, I.f32)
    within = I.reduce.sum(group_m2, axis=0, acc_dtype=I.f32)
    total_m2 = within + between
    denominator = I.cast(1048576, I.f32) - I.cast(correction, I.f32)
    denominator = I.maximum(denominator, I.cast(0.0, I.f32))
    output[()] = I.sqrt(total_m2 / denominator)


def build(context):
    group_stats = context.compile("std_group_stats", _std_group_stats)
    merge = context.compile("std_merge", _std_merge)
    workspace = None

    def std(
        input: torch.Tensor,
        dim=None,
        *,
        correction=1,
        keepdim=False,
        out=None,
    ) -> torch.Tensor:
        nonlocal workspace
        if workspace is None or workspace[0].device != input.device:
            workspace = (
                torch.empty((256,), device=input.device, dtype=torch.float32),
                torch.empty((256,), device=input.device, dtype=torch.float32),
            )
        group_means, group_m2 = workspace
        result = out
        if result is None:
            result = torch.empty((), device=input.device, dtype=input.dtype)
        group_stats(input, group_means, group_m2)
        merge(group_means, group_m2, result, correction)
        return result

    return std
