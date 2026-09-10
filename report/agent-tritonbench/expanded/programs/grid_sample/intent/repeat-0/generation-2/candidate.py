import intent
import intent.language as I


@intent.kernel
def grid_sample_kernel(
    input: I.In[I.f32, ("N", "C", "H", "W")],
    sample_grid: I.In[I.f32, ("N", "OH", "OW", 2)],
    output: I.Out[I.f32, ("N", "C", "OH", "OW")],
):
    batches = I.domain(0, input.shape[0])
    channels = I.domain(0, input.shape[1])
    output_rows = I.domain(0, sample_grid.shape[1])
    output_columns = I.domain(0, sample_grid.shape[2])

    height = I.cast(input.shape[2], I.f32)
    width = I.cast(input.shape[3], I.f32)

    for batch in I.parallel(batches):
        for channel in I.parallel(channels):
            for output_row in I.parallel(output_rows):
                for output_column in I.parallel(output_columns):
                    grid_x = sample_grid[batch, output_row, output_column, 0]
                    grid_y = sample_grid[batch, output_row, output_column, 1]

                    if grid_x != grid_x:
                        grid_x = -1.0
                    if grid_y != grid_y:
                        grid_y = -1.0

                    source_x = ((grid_x + 1.0) * width - 1.0) / 2.0
                    source_y = ((grid_y + 1.0) * height - 1.0) / 2.0

                    x0 = I.cast(source_x, I.index)
                    y0 = I.cast(source_y, I.index)
                    if I.cast(x0, I.f32) > source_x:
                        x0 = x0 - 1
                    if I.cast(y0, I.f32) > source_y:
                        y0 = y0 - 1
                    x1 = x0 + 1
                    y1 = y0 + 1

                    value_00 = 0.0
                    value_01 = 0.0
                    value_10 = 0.0
                    value_11 = 0.0

                    if y0 >= 0:
                        if y0 < input.shape[2]:
                            if x0 >= 0:
                                if x0 < input.shape[3]:
                                    value_00 = input[batch, channel, y0, x0]
                            if x1 >= 0:
                                if x1 < input.shape[3]:
                                    value_01 = input[batch, channel, y0, x1]
                    if y1 >= 0:
                        if y1 < input.shape[2]:
                            if x0 >= 0:
                                if x0 < input.shape[3]:
                                    value_10 = input[batch, channel, y1, x0]
                            if x1 >= 0:
                                if x1 < input.shape[3]:
                                    value_11 = input[batch, channel, y1, x1]

                    weight_x = source_x - I.cast(x0, I.f32)
                    weight_y = source_y - I.cast(y0, I.f32)
                    inverse_x = 1.0 - weight_x
                    inverse_y = 1.0 - weight_y

                    output[batch, channel, output_row, output_column] = (
                        value_00 * inverse_x * inverse_y
                        + value_01 * weight_x * inverse_y
                        + value_10 * inverse_x * weight_y
                        + value_11 * weight_x * weight_y
                    )


def build(context):
    compiled = context.compile("grid_sample_bilinear_zeros", grid_sample_kernel)

    def wrapper(
        input,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ):
        return compiled.run(input, grid)

    return wrapper
