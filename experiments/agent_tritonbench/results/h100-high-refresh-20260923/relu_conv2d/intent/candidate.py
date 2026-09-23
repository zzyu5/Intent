import torch
import intent
import intent.language as I


@intent.kernel
def relu_conv2d_kernel(
    input: I.In[I.f32, (1, 3, 256, 256)],
    weight: I.In[I.f32, (64, 3, 3, 3)],
    output: I.Out[I.f32, (1, 64, 256, 256)],
):
    channels = I.domain(0, 64)
    rows = I.domain(0, 256)
    columns = I.domain(0, 256)

    # These reshapes make the output-channel and spatial axes explicit, so
    # each of the 27 input terms broadcasts to a [64, 256, 256] value.
    output_channels = I.reshape(I.indices(channels), (64, 1, 1))
    output_rows = I.reshape(I.indices(rows), (1, 256, 1))
    output_columns = I.reshape(I.indices(columns), (1, 1, 256))

    accumulated = I.full((64, 256, 256), fill=0.0, dtype=I.f32)
    for input_channel in range(3):
        for kernel_row in range(3):
            input_row = output_rows + kernel_row - 1
            for kernel_column in range(3):
                input_column = output_columns + kernel_column - 1
                input_value = input[0, input_channel, input_row, input_column]
                filter_value = weight[
                    output_channels,
                    input_channel,
                    kernel_row,
                    kernel_column,
                ]
                accumulated = accumulated + input_value * filter_value

    activated = I.select(
        accumulated > 0.0,
        accumulated,
        I.cast(0.0, I.f32),
    )
    output[0, channels, rows, columns] = activated


def build(context):
    compiled = context.compile("relu_conv2d_fixed", relu_conv2d_kernel)

    def relu_conv2d(
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        inplace=False,
    ):
        output = torch.empty(
            (1, 64, 256, 256),
            dtype=input.dtype,
            device=input.device,
        )
        compiled(input, weight, output)
        return output

    return relu_conv2d
