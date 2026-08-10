import intent
import intent.language as I


CONV1D_BATCH = 64
CONV1D_LENGTH = 16384
CONV1D_FILTER = 5

CONV2D_BATCH = 16
CONV2D_HEIGHT = 256
CONV2D_WIDTH = 256
CONV2D_FILTER_HEIGHT = 3
CONV2D_FILTER_WIDTH = 3


@intent.kernel
def conv1d_same(
    x: I.In[I.f16, ("B", "L")],
    weight: I.In[I.f16, ("K",)],
    output: I.Out[I.f16, ("B", "L")],
):
    B, L = x.shape
    length = I.domain(0, L)
    taps = I.domain(0, CONV1D_FILTER)
    for batch in I.parallel(I.domain(0, B)):
        for output_region in I.parallel(
            I.partition(length, extent=I.auto("L_TILE"))
        ):
            output_indices = I.indices(output_region)
            tap_indices = I.indices(taps)
            input_indices = (
                output_indices[:, None]
                + tap_indices[None, :]
                - CONV1D_FILTER // 2
            )
            patch = I.cast(x[batch, input_indices], I.f32)
            filter_values = I.cast(weight[taps], I.f32)
            products = patch * filter_values
            reduced = I.reduce.sum(
                products,
                axis=1,
                identity=0.0,
                acc_dtype=I.f32,
            )
            output[batch, output_region] = I.cast(reduced, I.f16)


@intent.kernel
def conv2d_same(
    x: I.In[I.f16, ("B", "H", "W")],
    weight: I.In[I.f16, ("R", "S")],
    output: I.Out[I.f16, ("B", "H", "W")],
):
    B, H, W = x.shape
    height = I.domain(0, H)
    width = I.domain(0, W)
    kernel_rows = I.domain(0, CONV2D_FILTER_HEIGHT)
    kernel_columns = I.domain(0, CONV2D_FILTER_WIDTH)
    for batch in I.parallel(I.domain(0, B)):
        for output_rows in I.parallel(
            I.partition(height, extent=I.auto("H_TILE"))
        ):
            for output_columns in I.parallel(
                I.partition(width, extent=I.auto("W_TILE"))
            ):
                output_row_indices = I.reshape(
                    I.indices(output_rows),
                    (output_rows, 1, 1, 1),
                )
                output_column_indices = I.reshape(
                    I.indices(output_columns),
                    (1, output_columns, 1, 1),
                )
                kernel_row_indices = I.indices(kernel_rows)[:, None]
                kernel_column_indices = I.indices(kernel_columns)
                input_rows = (
                    output_row_indices
                    + kernel_row_indices
                    - CONV2D_FILTER_HEIGHT // 2
                )
                input_columns = (
                    output_column_indices
                    + kernel_column_indices
                    - CONV2D_FILTER_WIDTH // 2
                )
                patch = I.cast(x[batch, input_rows, input_columns], I.f32)
                filter_values = I.cast(
                    weight[kernel_rows, kernel_columns],
                    I.f32,
                )
                products = patch * filter_values
                reduced_columns = I.reduce.sum(
                    products,
                    axis=3,
                    identity=0.0,
                    acc_dtype=I.f32,
                )
                reduced = I.reduce.sum(
                    reduced_columns,
                    axis=2,
                    identity=0.0,
                    acc_dtype=I.f32,
                )
                output[batch, output_rows, output_columns] = I.cast(
                    reduced,
                    I.f16,
                )
