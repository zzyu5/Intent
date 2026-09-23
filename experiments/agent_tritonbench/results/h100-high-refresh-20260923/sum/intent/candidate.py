import torch
import intent
import intent.language as I


@intent.kernel
def sum_kernel(
    input: I.In[I.f32, (1048576,)],
    output: I.Out[I.f32, ()],
):
    values = input[I.domain(0, 1048576)]
    total = I.reduce.sum(values, axis=0, acc_dtype=I.f32)
    output[()] = total


def build(context):
    compiled_sum = context.compile("sum_kernel", sum_kernel)

    def sum(input, dim, keepdim=False, dtype=None):
        output = torch.empty((), device=input.device, dtype=torch.float32)
        compiled_sum(input, output)
        return output

    return sum
