import torch
import intent
import intent.language as I


@intent.kernel
def conv2d_kernel(
    input: I.In[I.f32, ("N", "C", "H", "W")],
    weight: I.In[I.f32, ("K", "C", "R", "S")],
    bias: I.In[I.f32, ("K",)],
    output: I.Out[I.f32, ("N", "K", "OH", "OW")],
):
    rows = I.domain(0, input.shape[2] - weight.shape[2] + 1)
    columns = I.domain(0, input.shape[3] - weight.shape[3] + 1)

    for n in I.parallel(I.domain(0, input.shape[0])):
        for k in I.parallel(I.domain(0, weight.shape[0])):
                for row in I.parallel(rows):
                    for column in I.parallel(columns):
                    channels = I.domain(0, weight.shape[1])
                    kernel_rows = I.domain(0, weight.shape[2])
                    kernel_columns = I.domain(0, weight.shape[3])
                    patch = input[
                        n,
                        channels,
                        row + kernel_rows,
                        column + kernel_columns,
                    ]
                    kernel = weight[k, channels, kernel_rows, kernel_columns]
                    value = bias[k] + I.contract(
                        patch,
                        kernel,
                        reduce=((0, 0), (1, 1), (2, 2)),
                        batch=(),
                        acc_dtype=I.f32,
                    )
                    output[n, k, row, column] = value


def build(context):
    compiled = context.compile("conv2d_fixed_profile", conv2d_kernel)

    def wrapper(
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    ):
        output = torch.empty(
            (
                input.shape[0],
                weight.shape[0],
                input.shape[2] - weight.shape[2] + 1,
                input.shape[3] - weight.shape[3] + 1,
            ),
            device=input.device,
            dtype=input.dtype,
        )
        compiled(input, weight, bias, output)
        return output

    return wrapper
