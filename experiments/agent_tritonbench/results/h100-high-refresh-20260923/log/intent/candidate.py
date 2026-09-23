import torch
import intent
import intent.language as I


@intent.kernel
def log_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    n, = input.shape
    for index in I.parallel(I.domain(0, n)):
        output[index] = I.log(input[index])


def build(context):
    compiled_log = context.compile("log_kernel", log_kernel)

    def log(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(input)
        compiled_log(input, out)
        return out

    return log
