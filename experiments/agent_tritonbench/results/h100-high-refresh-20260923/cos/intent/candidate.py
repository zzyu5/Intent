import torch
import intent
import intent.language as I


@intent.kernel
def cos_kernel(
    input_tensor: I.In[I.f32, (1048576,)],
    output: I.Out[I.f32, (1048576,)],
):
    elements = I.domain(0, 1048576)
    for index in I.parallel(elements):
        output[index] = I.cos(input_tensor[index])


def build(context):
    compiled_cos = context.compile("cos_kernel", cos_kernel)

    def cos(input_tensor):
        output = torch.empty_like(input_tensor)
        compiled_cos(input_tensor, output)
        return output

    return cos
