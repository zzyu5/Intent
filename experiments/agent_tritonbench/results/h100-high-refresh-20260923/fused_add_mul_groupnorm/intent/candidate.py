import torch
import intent
import intent.language as I


@intent.kernel
def _group_stats(
    input1: I.In[I.f32, (16, 64, 128, 64)],
    input2: I.In[I.f32, (16, 64, 128, 64)],
    mean_out: I.Out[I.f32, (16, 8)],
    var_out: I.Out[I.f32, (16, 8)],
):
    # Flatten each group for a compact reduction while retaining the logical
    # batch/group decomposition.
    x = I.reshape(input1, (16, 8, 65536))
    y = I.reshape(input2, (16, 8, 65536))
    fused = (x + y) * y

    group_size = I.cast(65536.0, I.f32)
    mean = I.reduce.sum(fused, axis=2, acc_dtype=I.f32) / group_size
    mean_expanded = I.reshape(mean, (16, 8, 1))
    centered = fused - mean_expanded
    variance = I.reduce.sum(centered * centered, axis=2, acc_dtype=I.f32) / group_size

    batches = I.domain(0, 16)
    groups = I.domain(0, 8)
    mean_out[batches, groups] = mean
    var_out[batches, groups] = variance


@intent.kernel
def _group_norm(
    input1: I.In[I.f32, (16, 64, 128, 64)],
    input2: I.In[I.f32, (16, 64, 128, 64)],
    weight: I.In[I.f32, (64,)],
    bias: I.In[I.f32, (64,)],
    mean: I.In[I.f32, (16, 8)],
    variance: I.In[I.f32, (16, 8)],
    eps: I.f32,
    output: I.Out[I.f32, (16, 64, 128, 64)],
):
    x = I.reshape(input1, (16, 8, 8, 128, 64))
    y = I.reshape(input2, (16, 8, 8, 128, 64))
    fused = (x + y) * y

    mean_expanded = I.reshape(mean, (16, 8, 1, 1, 1))
    variance_expanded = I.reshape(variance, (16, 8, 1, 1, 1))
    scale = I.reshape(weight, (1, 8, 8, 1, 1))
    shift = I.reshape(bias, (1, 8, 8, 1, 1))

    normalized = (fused - mean_expanded) / I.sqrt(variance_expanded + eps)
    result = normalized * scale + shift
    result = I.reshape(result, (16, 64, 128, 64))

    batches = I.domain(0, 16)
    channels = I.domain(0, 64)
    height = I.domain(0, 128)
    width = I.domain(0, 64)
    output[batches, channels, height, width] = result


def build(context):
    stats_kernel = context.compile("groupnorm_stats", _group_stats, constexprs={})
    norm_kernel = context.compile("groupnorm_apply", _group_norm, constexprs={})

    def fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups, eps=1e-5, *, out=None):
        if out is None:
            out = torch.empty_like(input1)

        mean = torch.empty((16, 8), device=input1.device, dtype=torch.float32)
        variance = torch.empty((16, 8), device=input1.device, dtype=torch.float32)

        stats_kernel(input1, input2, mean, variance)
        norm_kernel(input1, input2, weight, bias, mean, variance, eps, out)
        return out

    return fused_add_mul_groupnorm
