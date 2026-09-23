import torch
import intent
import intent.language as I


@intent.kernel
def mean_kernel(
    input_tensor: I.In[I.f32, (1048576,)],
    output: I.Out[I.f32, ()],
):
    elements = I.domain(0, 1048576)
    values = input_tensor[elements]
    total = I.reduce.sum(values, axis=0, acc_dtype=I.f32)
    output[()] = I.fdiv(total, I.cast(1048576, I.f32))


def build(context):
    mean_artifact = context.compile("mean_1d", mean_kernel)

    def mean(input_tensor, dim, keepdim=False, dtype=None, out=None):
        if out is None:
            output = torch.empty((), device=input_tensor.device, dtype=torch.float32)
        else:
            output = out
        mean_artifact(input_tensor, output)
        return output

    return mean
