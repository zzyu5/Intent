import torch
import intent
import intent.language as I


@intent.kernel
def _exp_mean_kernel(
    input: I.In[I.f32, (1048576,)],
    output: I.Out[I.f32, ()],
):
    values = input[I.domain(0, 1048576)]
    exp_values = I.exp(values)
    total = I.reduce.sum(exp_values, axis=0, acc_dtype=I.f32)
    output[()] = I.fdiv(total, I.cast(1048576, I.f32))


def build(context):
    kernel = context.compile("exp_mean", _exp_mean_kernel)

    def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None):
        result = out
        if result is None:
            result = torch.empty((), device=input.device, dtype=torch.float32)
        kernel(input, result)
        return result

    return exp_mean
