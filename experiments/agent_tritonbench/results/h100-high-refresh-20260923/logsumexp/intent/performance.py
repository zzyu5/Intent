import torch
import intent
import intent.language as I


GROUPS = 4096
GROUP_SIZE = 256


@intent.kernel
def grouped_logsumexp_stage(
    x: I.In[I.f32, (4096, 256)],
    group_max: I.Out[I.f32, (4096,)],
    group_sum: I.Out[I.f32, (4096,)],
):
    groups = I.domain(0, 4096)
    members = I.domain(0, 256)
    values = x[groups, members]

    local_max = I.reduce.max(values, axis=1, acc_dtype=I.f32)
    local_max_2d = I.reshape(local_max, (4096, 1))

    # Avoid inf - inf before the exponential for all-infinite groups.
    zero = I.cast(0.0, I.f32)
    delta = I.select(values == local_max_2d, zero, values - local_max_2d)
    local_sum = I.reduce.sum(I.exp(delta), axis=1, acc_dtype=I.f32)

    group_max[groups] = local_max
    group_sum[groups] = local_sum


@intent.kernel
def finish_logsumexp(
    group_max: I.In[I.f32, (4096,)],
    group_sum: I.In[I.f32, (4096,)],
    output: I.Out[I.f32, ()],
):
    groups = I.domain(0, 4096)
    global_max = I.reduce.max(group_max[groups], axis=0, acc_dtype=I.f32)

    zero = I.cast(0.0, I.f32)
    global_delta = I.select(
        group_max[groups] == global_max,
        zero,
        group_max[groups] - global_max,
    )
    scaled = group_sum[groups] * I.exp(global_delta)
    total = I.reduce.sum(scaled, axis=0, acc_dtype=I.f32)
    result = global_max + I.log(total)
    output[()] = result


def build(context):
    stage_artifact = context.compile("grouped_logsumexp_stage", grouped_logsumexp_stage)
    finish_artifact = context.compile("finish_logsumexp", finish_logsumexp)

    def logsumexp(input, dim, keepdim=False, *, out=None):
        # The task profile is a contiguous 1D reduction over dim 0.
        grouped_input = input.reshape(GROUPS, GROUP_SIZE)
        group_max = torch.empty((GROUPS,), device=input.device, dtype=input.dtype)
        group_sum = torch.empty((GROUPS,), device=input.device, dtype=input.dtype)

        stage_artifact(grouped_input, group_max, group_sum)

        if out is None:
            result = torch.empty((1,) if keepdim else (), device=input.device, dtype=input.dtype)
        else:
            result = out
        scalar_result = result.reshape(())
        finish_artifact(group_max, group_sum, scalar_result)
        return result

    return logsumexp
