import intent
import intent.language as I


BATCH = 8
CHANNELS = 32
HEIGHT = 128
WIDTH = 128
KERNEL_HEIGHT = 3
KERNEL_WIDTH = 3
STRIDE = 2
PADDING = 1
OUTPUT_HEIGHT = (HEIGHT + 2 * PADDING - KERNEL_HEIGHT) // STRIDE + 1
OUTPUT_WIDTH = (WIDTH + 2 * PADDING - KERNEL_WIDTH) // STRIDE + 1


@intent.kernel
def max_pool2d(
    x: I.In[I.f16, (BATCH, CHANNELS, HEIGHT, WIDTH)],
    output: I.Out[I.f16, (BATCH, CHANNELS, OUTPUT_HEIGHT, OUTPUT_WIDTH)],
):
    output_rows = I.domain(0, OUTPUT_HEIGHT)
    output_columns = I.domain(0, OUTPUT_WIDTH)
    kernel_rows = I.domain(0, KERNEL_HEIGHT)
    kernel_columns = I.domain(0, KERNEL_WIDTH)
    for batch in I.parallel(I.domain(0, BATCH)):
        for channel in I.parallel(I.domain(0, CHANNELS)):
            row_indices = I.reshape(
                I.indices(output_rows) * STRIDE - PADDING,
                (output_rows, 1, 1, 1),
            )
            column_indices = I.reshape(
                I.indices(output_columns) * STRIDE - PADDING,
                (1, output_columns, 1, 1),
            )
            input_rows = row_indices + I.indices(kernel_rows)[:, None]
            input_columns = column_indices + I.indices(kernel_columns)
            patch = x[batch, channel, input_rows, input_columns]
            column_maxima = I.reduce.max(
                patch,
                axis=3,
                identity=-I.inf,
            )
            maxima = I.reduce.max(
                column_maxima,
                axis=2,
                identity=-I.inf,
            )
            output[batch, channel, output_rows, output_columns] = maxima
