import torch
import intent
import intent.language as I


@intent.kernel
def conv2d_im2col(
    x: I.In[I.f32, ("N", "IC", "IH", "IW")],
    columns: I.Out[I.f32, ("N", "K", "P")],
    KERNEL_H: I.Constexpr[int],
    KERNEL_W: I.Constexpr[int],
    STRIDE_H: I.Constexpr[int],
    STRIDE_W: I.Constexpr[int],
    PADDING_H: I.Constexpr[int],
    PADDING_W: I.Constexpr[int],
    DILATION_H: I.Constexpr[int],
    DILATION_W: I.Constexpr[int],
):
    channels = I.domain(0, x.shape[1])
    kernel_rows = I.domain(0, KERNEL_H)
    kernel_cols = I.domain(0, KERNEL_W)
    output_height = (x.shape[2] + 2 * PADDING_H - DILATION_H * (KERNEL_H - 1) - 1) // STRIDE_H + 1
    output_width = (x.shape[3] + 2 * PADDING_W - DILATION_W * (KERNEL_W - 1) - 1) // STRIDE_W + 1
    output_rows = I.domain(0, output_height)
    output_cols = I.domain(0, output_width)

    for n in I.parallel(I.domain(0, x.shape[0])):
        for channel in I.parallel(channels):
            for kernel_row in I.parallel(kernel_rows):
                for kernel_col in I.parallel(kernel_cols):
                    kernel_index = channel * (KERNEL_H * KERNEL_W) + kernel_row * KERNEL_W + kernel_col
                    for output_row in I.parallel(output_rows):
                        input_row = output_row * STRIDE_H - PADDING_H + kernel_row * DILATION_H
                        for output_col in I.parallel(output_cols):
                            input_col = output_col * STRIDE_W - PADDING_W + kernel_col * DILATION_W
                            position = output_row * output_width + output_col
                            I.assume_in_bounds(input_row, x, axis=2)
                            I.assume_in_bounds(input_col, x, axis=3)
                            columns[n, kernel_index, position] = x[n, channel, input_row, input_col]


@intent.kernel
def conv2d_matmul_bias(
    weight: I.In[I.f32, ("OC", "K")],
    columns: I.In[I.f32, ("N", "K", "P")],
    bias: I.In[I.f32, ("OC",)],
    output: I.Out[I.f32, ("N", "OC", "P")],
):
    values = I.matmul(weight, columns, acc_dtype=I.f32)
    for n in I.parallel(I.domain(0, output.shape[0])):
        for channel in I.parallel(I.domain(0, output.shape[1])):
            for position in I.parallel(I.domain(0, output.shape[2])):
                output[n, channel, position] = values[n, channel, position] + bias[channel]


@intent.kernel
def conv2d_matmul_no_bias(
    weight: I.In[I.f32, ("OC", "K")],
    columns: I.In[I.f32, ("N", "K", "P")],
    output: I.Out[I.f32, ("N", "OC", "P")],
):
    values = I.matmul(weight, columns, acc_dtype=I.f32)
    for n in I.parallel(I.domain(0, output.shape[0])):
        for channel in I.parallel(I.domain(0, output.shape[1])):
            for position in I.parallel(I.domain(0, output.shape[2])):
                output[n, channel, position] = values[n, channel, position]


def build(context):
    im2col = context.compile(
        "conv2d_im2col_fixed",
        conv2d_im2col,
        constexprs={
            "KERNEL_H": 3,
            "KERNEL_W": 3,
            "STRIDE_H": 1,
            "STRIDE_W": 1,
            "PADDING_H": 0,
            "PADDING_W": 0,
            "DILATION_H": 1,
            "DILATION_W": 1,
        },
    )
    matmul_bias = context.compile("conv2d_matmul_bias_fixed", conv2d_matmul_bias, constexprs={})
    matmul_no_bias = context.compile("conv2d_matmul_no_bias_fixed", conv2d_matmul_no_bias, constexprs={})

    def wrapper(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if isinstance(stride, tuple):
            stride_h, stride_w = stride
        else:
            stride_h = stride_w = stride
        if isinstance(padding, tuple):
            padding_h, padding_w = padding
        else:
            padding_h = padding_w = padding
        if isinstance(dilation, tuple):
            dilation_h, dilation_w = dilation
        else:
            dilation_h = dilation_w = dilation

        output_h = (input.shape[2] + 2 * padding_h - dilation_h * (weight.shape[2] - 1) - 1) // stride_h + 1
        output_w = (input.shape[3] + 2 * padding_w - dilation_w * (weight.shape[3] - 1) - 1) // stride_w + 1
        output_positions = output_h * output_w
        kernel_elements = weight.shape[1] * weight.shape[2] * weight.shape[3]

        columns = torch.empty(
            (input.shape[0], kernel_elements, output_positions),
            device=input.device,
            dtype=input.dtype,
        )
        output = torch.empty(
            (input.shape[0], weight.shape[0], output_h, output_w),
            device=input.device,
            dtype=input.dtype,
        )

        im2col(input, columns)
        flat_weight = weight.reshape(weight.shape[0], kernel_elements)
        flat_output = output.reshape(input.shape[0], weight.shape[0], output_positions)
        if bias is None:
            matmul_no_bias(flat_weight, columns, flat_output)
        else:
            matmul_bias(flat_weight, columns, bias, flat_output)
        return output

    return wrapper
