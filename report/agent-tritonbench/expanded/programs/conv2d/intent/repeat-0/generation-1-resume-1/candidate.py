import intent
import intent.language as I


@intent.kernel
def conv2d_kernel(
    input: I.In[I.f32, (8, 64, 16, 16)],
    weight: I.In[I.f32, (128, 64, 3, 3)],
    bias: I.In[I.f32, (128,)],
    output: I.Out[I.f32, (8, 128, 14, 14)],
    STRIDE: I.Constexpr[int],
    PADDING: I.Constexpr[int],
    DILATION: I.Constexpr[int],
    GROUPS: I.Constexpr[int],
):
    batches = I.domain(0, input.shape[0])
    output_channels = I.domain(0, weight.shape[0])
    output_rows = I.domain(0, output.shape[2])
    output_columns = I.domain(0, output.shape[3])
    input_channels = I.domain(0, input.shape[1] // GROUPS)
    kernel_rows = I.domain(0, weight.shape[2])
    kernel_columns = I.domain(0, weight.shape[3])

    for batch in I.parallel(batches):
        for channel in I.parallel(output_channels):
            for row in I.parallel(output_rows):
                for column in I.parallel(output_columns):
                    value = I.cast(0, I.f32)
                    for input_channel in input_channels:
                        for kernel_row in kernel_rows:
                            for kernel_column in kernel_columns:
                                input_row = (
                                    row * STRIDE
                                    + kernel_row * DILATION
                                    - PADDING
                                )
                                input_column = (
                                    column * STRIDE
                                    + kernel_column * DILATION
                                    - PADDING
                                )
                                value = value + (
                                    input[
                                        batch,
                                        input_channel,
                                        input_row,
                                        input_column,
                                    ]
                                    * weight[
                                        channel,
                                        input_channel,
                                        kernel_row,
                                        kernel_column,
                                    ]
                                )
                    output[batch, channel, row, column] = value + bias[channel]


def build(context):
    compiled = context.compile(
        "conv2d_fixed_nchw",
        conv2d_kernel,
        constexprs={
            "STRIDE": 1,
            "PADDING": 0,
            "DILATION": 1,
            "GROUPS": 1,
        },
    )

    def wrapper(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return compiled.run(input, weight, bias)

    return wrapper
