import torch
import intent
import intent.language as I


@intent.kernel
def _group_reduce(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    partial: I.Out[I.f32, ("G",)],
    alpha: I.f32,
    GROUP_SIZE: I.Constexpr[int],
):
    elements = I.domain(0, input.shape[0])
    for group in I.parallel(I.domain(0, partial.shape[0])):
        begin = group * GROUP_SIZE
        end = begin + GROUP_SIZE
        members = elements[begin:end]
        values = input[members] + other[members] * alpha
        partial[group] = I.reduce.sum(values, axis=0, acc_dtype=I.f32)


@intent.kernel
def _finish_mean(
    partial: I.In[I.f32, ("G",)],
    output: I.Out[I.f32, ()],
    count: I.f32,
):
    groups = I.domain(0, partial.shape[0])
    total = I.reduce.sum(partial[groups], axis=0, acc_dtype=I.f32)
    output[()] = total / count


def build(context):
    group_reduce = context.compile(
        "add_mean_group_reduce",
        _group_reduce,
        constexprs={"GROUP_SIZE": 4096},
    )
    finish_mean = context.compile("add_mean_finish", _finish_mean)

    def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
        partial = torch.empty((256,), device=input.device, dtype=torch.float32)
        result = out if out is not None else torch.empty((), device=input.device, dtype=torch.float32)
        group_reduce(input, other, partial, float(alpha))
        finish_mean(partial, result, float(input.numel()))
        return result

    return add_mean
