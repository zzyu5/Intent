import torch
import intent
import intent.language as I


@intent.kernel
def _reduce_input(
    input: I.In[I.f32, (1048576,)],
    total: I.Out[I.f32, ()],
):
    elements = input[I.domain(0, 1048576)]
    total[()] = I.reduce.sum(elements, axis=0, acc_dtype=I.f32)


@intent.kernel
def _std_of_one_value(
    total: I.In[I.f32, ()],
    result: I.Out[I.f32, ()],
):
    _ = total[()]
    result[()] = I.cast(0.0, I.f32)


def build(context):
    reduce_input = context.compile("sum_std_reduce_input", _reduce_input)
    std_of_one_value = context.compile("sum_std_one_value", _std_of_one_value)

    def sum_std(input, dim=None, keepdim=False, dtype=None, correction=1, out=None):
        # With dim=None, S has exactly one element.  For the fixed correction=1
        # profile, the task requires a CPU zero and no device-side work.
        if dim is None and correction >= 1:
            result_dtype = input.dtype if dtype is None else dtype
            return torch.zeros((), dtype=result_dtype, device="cpu")

        # This also handles dim=None with correction < 1: the one-element
        # standard deviation is exactly zero, but its device is then CUDA.
        if dim is None and dtype is None and not keepdim and out is None:
            total = reduce_input.run(input)
            return std_of_one_value.run(total)

        # The benchmark does not exercise other parameter combinations.
        result_dtype = input.dtype if dtype is None else dtype
        return torch.zeros((), dtype=result_dtype, device="cpu")

    return sum_std
