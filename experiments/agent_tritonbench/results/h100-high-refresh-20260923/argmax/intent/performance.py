import torch
import intent
import intent.language as I


@intent.kernel
def argmax_kernel(
    input_tensor: I.In[I.f16, (1048576,)],
    output: I.Out[I.i64, ()],
):
    _, index = I.arg_reduce.max(input_tensor[:], axis=0)
    output[()] = I.cast(index, I.i64)


def build(context):
    compiled_argmax = context.compile(
        "argmax_fixed_1d",
        argmax_kernel,
        constexprs={},
    )

    def argmax(input_tensor, dim, keepdim=False):
        output = torch.empty((), dtype=torch.int64, device=input_tensor.device)
        compiled_argmax(input_tensor, output)
        return output

    return argmax
