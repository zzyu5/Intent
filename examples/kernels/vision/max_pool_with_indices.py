import intent
import intent.language as I

from .max_pool import BATCH
from .max_pool import CHANNELS
from .max_pool import HEIGHT
from .max_pool import OUTPUT_HEIGHT
from .max_pool import OUTPUT_WIDTH
from .max_pool import PADDING
from .max_pool import STRIDE
from .max_pool import WIDTH


KERNEL_HEIGHT = 3
KERNEL_WIDTH = 3
KERNEL_ELEMENTS = KERNEL_HEIGHT * KERNEL_WIDTH


@intent.kernel
def max_pool2d_with_indices(
    x: I.In[I.f16, (BATCH, CHANNELS, HEIGHT, WIDTH)],
    output: I.Out[I.f16, (BATCH, CHANNELS, OUTPUT_HEIGHT, OUTPUT_WIDTH)],
    indices: I.Out[I.i64, (BATCH, CHANNELS, OUTPUT_HEIGHT, OUTPUT_WIDTH)],
):
    output_rows = I.domain(0, OUTPUT_HEIGHT)
    output_columns = I.domain(0, OUTPUT_WIDTH)
    filter_elements = I.domain(0, KERNEL_ELEMENTS)
    for batch in I.parallel(I.domain(0, BATCH)):
        for channel in I.parallel(I.domain(0, CHANNELS)):
            for row_region in I.parallel(
                I.partition(output_rows, extent=I.auto("H_TILE"))
            ):
                for column_region in I.parallel(
                    I.partition(output_columns, extent=I.auto("W_TILE"))
                ):
                    local_index = I.indices(filter_elements)
                    filter_row = local_index // KERNEL_WIDTH
                    filter_column = local_index % KERNEL_WIDTH
                    input_rows = I.reshape(
                        I.indices(row_region) * STRIDE - PADDING,
                        (row_region, 1, 1),
                    ) + I.reshape(filter_row, (1, 1, filter_elements))
                    input_columns = I.reshape(
                        I.indices(column_region) * STRIDE - PADDING,
                        (1, column_region, 1),
                    ) + I.reshape(filter_column, (1, 1, filter_elements))
                    patch = x[batch, channel, input_rows, input_columns]
                    maximum, winner = I.arg_reduce.max(
                        patch,
                        axis=2,
                        identity=-I.inf,
                    )
                    winner_index = I.cast(winner, I.index)
                    winner_row = winner_index // KERNEL_WIDTH
                    winner_column = winner_index % KERNEL_WIDTH
                    global_row = (
                        I.indices(row_region)[:, None] * STRIDE
                        - PADDING
                        + winner_row
                    )
                    global_column = (
                        I.indices(column_region)[None, :] * STRIDE
                        - PADDING
                        + winner_column
                    )
                    output[batch, channel, row_region, column_region] = maximum
                    indices[batch, channel, row_region, column_region] = I.cast(
                        global_row * WIDTH + global_column,
                        I.i64,
                    )
