import torch
import intent
import intent.language as I


@intent.kernel
def _sigmoid_argmax_kernel(
    input: I.In[I.f32, (1024, 1024)],
    output: I.Out[I.i64, (1024,)],
):
    rows = I.domain(0, 1024)
    columns = I.domain(0, 1024)
    values = input[rows, columns]

    one = I.cast(1.0, I.f32)
    sigmoid_values = I.fdiv(one, one + I.exp(-values))
    _, indices = I.arg_reduce.max(sigmoid_values, axis=1)
    output[rows] = I.cast(indices, I.i64)


def build(context):
    kernel = context.compile("sigmoid_argmax", _sigmoid_argmax_kernel, constexprs={})

    def sigmoid_argmax(input, dim=None, keepdim=False):
        output = torch.empty((input.shape[0],), dtype=torch.int64, device=input.device)
        kernel(input, output)
        return output

    return sigmoid_argmax
