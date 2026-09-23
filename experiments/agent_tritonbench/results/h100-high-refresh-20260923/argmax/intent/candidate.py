import torch
import intent
import intent.language as I


N = 1048576


@intent.kernel
def argmax_scalar_kernel(
    input_tensor: I.In[I.f16, (N,)],
    output: I.Out[I.i64, ()],
):
    axis = I.domain(0, 1048576)
    values = input_tensor[axis]
    _, index = I.arg_reduce.max(values, axis=0)
    output[()] = I.cast(index, I.i64)


@intent.kernel
def argmax_keepdim_kernel(
    input_tensor: I.In[I.f16, (N,)],
    output: I.Out[I.i64, (1,)],
):
    axis = I.domain(0, 1048576)
    values = input_tensor[axis]
    _, index = I.arg_reduce.max(values, axis=0)
    output[0] = I.cast(index, I.i64)


def build(context):
    scalar_artifact = context.compile("argmax_scalar", argmax_scalar_kernel)
    keepdim_artifact = context.compile("argmax_keepdim", argmax_keepdim_kernel)

    def argmax(input_tensor, dim, keepdim=False):
        if dim not in (None, 0, -1):
            raise ValueError("only the fixed one-dimensional profile is supported")
        if keepdim:
            output = torch.empty((1,), dtype=torch.int64, device=input_tensor.device)
            keepdim_artifact(input_tensor, output)
        else:
            output = torch.empty((), dtype=torch.int64, device=input_tensor.device)
            scalar_artifact(input_tensor, output)
        return output

    return argmax
