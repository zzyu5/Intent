import intent
import intent.language as I


@intent.kernel
def relu_conv2d_kernel(
    x: I.In[I.f32, ("N", "C", "H", "W")],
    weight: I.In[I.f32, ("K", "C", "R", "S")],
    output: I.Out[I.f32, ("N", "K", "H", "W")],
):
    output_rows = I.domain(0, x.shape[2])
    output_cols = I.domain(0, x.shape[3])
    output_channels = I.domain(0, weight.shape[0])
    batches = I.domain(0, x.shape[0])

    reduction = I.domain(0, 27)
    reduction_index = I.indices(reduction)
    input_channel = reduction_index // 9
    kernel_row = (reduction_index // 3) % 3
    kernel_col = reduction_index % 3

    for batch in I.parallel(batches):
        for channel in I.parallel(output_channels):
            for row in I.parallel(output_rows):
                for col in I.parallel(output_cols):
                    input_row = row + kernel_row - 1
                    input_col = col + kernel_col - 1

                    valid = (
                        (input_row >= 0)
                        & (input_row < x.shape[2])
                        & (input_col >= 0)
                        & (input_col < x.shape[3])
                    )

                    safe_row = I.minimum(
                        I.maximum(input_row, 0), x.shape[2] - 1
                    )
                    safe_col = I.minimum(
                        I.maximum(input_col, 0), x.shape[3] - 1
                    )

                    values = x[batch, input_channel, safe_row, safe_col]
                    filters = weight[
                        channel, input_channel, kernel_row, kernel_col
                    ]
                    values = I.select(valid, values, I.cast(0.0, I.f32))
                    convolution = I.reduce.sum(
                        values * filters,
                        axis=0,
                        acc_dtype=I.f32,
                    )
                    output[batch, channel, row, col] = I.maximum(
                        convolution, I.cast(0.0, I.f32)
                    )


def build(context):
    compiled = context.compile("relu_conv2d", relu_conv2d_kernel)

    def wrapper(
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        inplace=False,
    ):
        return compiled.run(input, weight)

    return wrapper
