import torch
import intent
import intent.language as I


GROUPS = 1024
CHUNK = 1024
ELEMENTS = GROUPS * CHUNK


@intent.kernel
def exp_partial(
    x: I.In[I.f32, ("G", "C")],
    partial: I.Out[I.f32, ("G",)],
):
    groups = I.domain(0, 1024)
    columns = I.domain(0, 1024)
    for group in I.parallel(groups):
        values = I.exp(x[group, columns])
        partial[group] = I.reduce.sum(values, axis=0)


@intent.kernel
def finish_mean(
    partial: I.In[I.f32, ("G",)],
    output: I.Out[I.f32, ()],
):
    groups = I.domain(0, 1024)
    total = I.reduce.sum(partial[groups], axis=0)
    output[()] = I.fdiv(total, I.cast(1048576, I.f32))


def build(context):
    partial_artifact = context.compile("exp_mean_partial", exp_partial)
    finish_artifact = context.compile("exp_mean_finish", finish_mean)

    def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None):
        x = input.reshape(GROUPS, CHUNK)
        partial = torch.empty((GROUPS,), device=input.device, dtype=torch.float32)
        result = out if out is not None else torch.empty((), device=input.device, dtype=torch.float32)
        partial_artifact(x, partial)
        finish_artifact(partial, result)
        return result

    return exp_mean
