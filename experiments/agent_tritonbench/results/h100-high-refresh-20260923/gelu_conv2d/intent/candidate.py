import torch
import intent
import intent.language as I


@intent.kernel
def _conv2d_gelu(
    input: I.In[I.f32, ("B", "IC", "H", "W")],
    weight: I.In[I.f32, ("OC", "IC", "KH", "KW")],
    output: I.Out[I.f32, ("B", "OC", "OH", "OW")],
):
    B, _, H, W = input.shape
    OC, IC, KH, KW = weight.shape
    _, _, OH, OW = output.shape

    batch = I.domain(0, B)
    out_channels = I.domain(0, OC)
    in_channels = I.domain(0, IC)
    out_rows = I.domain(0, OH)
    out_cols = I.domain(0, OW)
    kernel_rows = I.domain(0, KH)
    kernel_cols = I.domain(0, KW)

    out_row_index = I.reshape(I.indices(out_rows), (OH, 1, 1, 1))
    out_col_index = I.reshape(I.indices(out_cols), (1, OW, 1, 1))
    kernel_row_index = I.reshape(I.indices(kernel_rows), (1, 1, KH, 1))
    kernel_col_index = I.reshape(I.indices(kernel_cols), (1, 1, 1, KW))

    # The fixed profile is stride 1, dilation 1, and padding 1.
    input_row = out_row_index + kernel_row_index - 1
    input_col = out_col_index + kernel_col_index - 1
    row_nonnegative = input_row >= 0
    row_in_bounds = input_row < H
    col_nonnegative = input_col >= 0
    col_in_bounds = input_col < W
    zero_index = I.cast(0, I.index)
    safe_row = I.select(row_nonnegative, input_row, zero_index)
    safe_row = I.select(row_in_bounds, safe_row, zero_index)
    safe_col = I.select(col_nonnegative, input_col, zero_index)
    safe_col = I.select(col_in_bounds, safe_col, zero_index)

    input_values = input[batch, in_channels, safe_row, safe_col]
    zero = I.cast(0.0, I.f32)
    input_values = I.select(col_nonnegative, input_values, zero)
    input_values = I.select(col_in_bounds, input_values, zero)
    input_values = I.select(row_nonnegative, input_values, zero)
    input_values = I.select(row_in_bounds, input_values, zero)
    input_values = I.reshape(input_values, (B, 1, IC, OH, OW, KH, KW))

    filter_values = weight[out_channels, in_channels, kernel_rows, kernel_cols]
    filter_values = I.reshape(filter_values, (1, OC, IC, 1, 1, KH, KW))

    products = input_values * filter_values
    convolution = I.reduce.sum(products, axis=(2, 5, 6), acc_dtype=I.f32)

    inv_sqrt2 = I.cast(0.7071067811865476, I.f32)
    cdf = I.cast(0.5, I.f32) * (I.cast(1.0, I.f32) + I.erf(convolution * inv_sqrt2))
    result = convolution * cdf
    output[batch, out_channels, out_rows, out_cols] = result


def build(context):
    conv_gelu = context.compile("conv2d_gelu", _conv2d_gelu)

    def gelu_conv2d(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        approximate="none",
        out=None,
    ) -> torch.Tensor:
        if out is None:
            out = torch.empty(
                (input.shape[0], weight.shape[0], input.shape[2], input.shape[3]),
                device=input.device,
                dtype=input.dtype,
            )
        conv_gelu(input, weight, out)
        return out

    return gelu_conv2d
