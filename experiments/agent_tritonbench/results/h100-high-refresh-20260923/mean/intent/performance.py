import torch
import intent
import intent.language as I


GROUPS = 1024
CHUNK = 1024
ELEMENTS = GROUPS * CHUNK


@intent.kernel
def mean_partials(
    input_tensor: I.In[I.f32, (ELEMENTS,)],
    partials: I.Out[I.f32, (GROUPS,)],
):
    elements = I.domain(0, 1048576)
    groups = I.domain(0, 1024)
    for group in I.parallel(groups):
        begin = group * 1024
        segment = elements[begin : begin + 1024]
        values = input_tensor[segment]
        partials[group] = I.reduce.sum(values, axis=0, acc_dtype=I.f32)


@intent.kernel
def mean_finish(
    partials: I.In[I.f32, (GROUPS,)],
    output: I.Out[I.f32, ()],
):
    groups = I.domain(0, 1024)
    total = I.reduce.sum(partials[groups], axis=0, acc_dtype=I.f32)
    output[()] = I.fdiv(total, I.cast(1048576, I.f32))


def build(context):
    partials_artifact = context.compile("mean_partials", mean_partials, constexprs={})
    finish_artifact = context.compile("mean_finish", mean_finish, constexprs={})

    def mean(input_tensor, dim, keepdim=False, dtype=None, out=None):
        if out is None:
            output = torch.empty((), device=input_tensor.device, dtype=torch.float32)
        else:
            output = out
        partials = torch.empty((GROUPS,), device=input_tensor.device, dtype=torch.float32)
        partials_artifact(input_tensor, partials)
        finish_artifact(partials, output)
        return output

    return mean
