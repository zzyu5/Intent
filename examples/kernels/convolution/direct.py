import intent
import intent.language as I


CONV1D_BATCH = 64
CONV1D_LENGTH = 16384
CONV1D_FILTER = 5
CAUSAL_CONV_BATCH = 8
CAUSAL_CONV_CHANNELS = 2048
CAUSAL_CONV_LENGTH = 4096
CAUSAL_CONV_WIDTH = 4
CAUSAL_UPDATE_BATCH = 64
CAUSAL_UPDATE_CHANNELS = 4096

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
def causal_depthwise_conv1d(
    x: I.In[I.f16, ("B", "D", "L")],
    weight: I.In[I.f16, ("D", "W")],
    bias: I.In[I.f16, ("D",)],
    output: I.Out[I.f16, ("B", "D", "L")],
    SILU: I.Constexpr[bool],
):
    B, D, L = x.shape
    W = weight.shape[1]
    positions = I.domain(0, L)
    taps = I.domain(0, W)
    for batch in I.parallel(I.domain(0, B)):
        for channel in I.parallel(I.domain(0, D)):
            for output_region in I.parallel(
                I.partition(positions, extent=I.auto("L_TILE"))
            ):
                output_index = I.indices(output_region)[:, None]
                tap_index = I.indices(taps)[None, :]
                input_index = output_index - (W - 1) + tap_index
                valid = input_index >= 0
                patch = x[batch, channel, input_index]
                patch = I.mask(
                    patch,
                    valid=valid,
                    fill=I.cast(0.0, I.f16),
                )
                products = (
                    I.cast(patch, I.f32)
                    * I.cast(weight[channel, taps], I.f32)[None, :]
                )
                reduced = I.reduce.sum(
                    products,
                    axis=1,
                    identity=0.0,
                    acc_dtype=I.f32,
                ) + I.cast(bias[channel], I.f32)
                if SILU:
                    reduced = reduced * I.sigmoid(reduced)
                output[batch, channel, output_region] = I.cast(reduced, I.f16)


@intent.kernel
def causal_depthwise_conv1d_update(
    x: I.In[I.f16, ("B", "D")],
    state: I.InOut[I.f16, ("B", "D", CAUSAL_CONV_WIDTH)],
    weight: I.In[I.f16, ("D", CAUSAL_CONV_WIDTH)],
    bias: I.In[I.f16, ("D",)],
    output: I.Out[I.f16, ("B", "D")],
    SILU: I.Constexpr[bool],
):
    B, D = x.shape
    channels = I.domain(0, D)
    for batch in I.parallel(I.domain(0, B)):
        for channel_region in I.parallel(
            I.partition(channels, extent=I.auto("D_TILE"))
        ):
            accumulator = I.cast(bias[channel_region], I.f32)
            for tap in range(CAUSAL_CONV_WIDTH - 1):
                shifted = state[batch, channel_region, tap + 1]
                state[batch, channel_region, tap] = shifted
                accumulator = accumulator + I.cast(shifted, I.f32) * I.cast(
                    weight[channel_region, tap],
                    I.f32,
                )
            current = x[batch, channel_region]
            state[batch, channel_region, CAUSAL_CONV_WIDTH - 1] = current
            accumulator = accumulator + I.cast(current, I.f32) * I.cast(
                weight[channel_region, CAUSAL_CONV_WIDTH - 1],
                I.f32,
            )
            if SILU:
                accumulator = accumulator * I.sigmoid(accumulator)
            output[batch, channel_region] = I.cast(accumulator, I.f16)


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
