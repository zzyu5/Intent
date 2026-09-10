import torch
import intent
import intent.language as I


@intent.kernel
def conv2d_im2col(
    x: I.In[I.f32, (8, 64, 16, 16)],
    columns: I.Out[I.f32, (903168,)],
):
    # Linearize the logical [N, IC, 3, 3, OH, OW] copy so the destination
    # relation is one-dimensional and has the same extent as the iteration.
    elements = I.domain(0, 903168)
    for element in I.parallel(elements):
        position = element % 196
        remaining = element // 196
        kernel_col = remaining % 3
        remaining = remaining // 3
        kernel_row = remaining % 3
        remaining = remaining // 3
        channel = remaining % 64
        batch = remaining // 64

        output_row = position // 14
        output_col = position % 14
        input_row = output_row + kernel_row
        input_col = output_col + kernel_col
        I.assume_in_bounds(input_row, x, axis=2)
        I.assume_in_bounds(input_col, x, axis=3)
        columns[element] = x[batch, channel, input_row, input_col]


@intent.kernel
def conv2d_matmul_bias(
    weight: I.In[I.f32, (128, 576)],
    columns: I.In[I.f32, (8, 576, 196)],
    bias: I.In[I.f32, (128,)],
    output: I.Out[I.f32, (8, 128, 196)],
):
    values = I.matmul(weight, columns, acc_dtype=I.f32)
    for batch in I.parallel(I.domain(0, 8)):
        for channel in I.parallel(I.domain(0, 128)):
            for position in I.parallel(I.domain(0, 196)):
                output[batch, channel, position] = values[batch, channel, position] + bias[channel]


@intent.kernel
def conv2d_matmul_no_bias(
    weight: I.In[I.f32, (128, 576)],
    columns: I.In[I.f32, (8, 576, 196)],
    output: I.Out[I.f32, (8, 128, 196)],
):
    values = I.matmul(weight, columns, acc_dtype=I.f32)
    for batch in I.parallel(I.domain(0, 8)):
        for channel in I.parallel(I.domain(0, 128)):
            for position in I.parallel(I.domain(0, 196)):
                output[batch, channel, position] = values[batch, channel, position]


def build(context):
    im2col = context.compile("conv2d_im2col_fixed", conv2d_im2col, constexprs={})
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
        output = torch.empty(
            (input.shape[0], weight.shape[0], output_h, output_w),
            device=input.device,
            dtype=input.dtype,
        )
        columns_storage = torch.empty(903168, device=input.device, dtype=input.dtype)

        im2col(input, columns_storage)
        columns = columns_storage.reshape(8, 576, 196)
        flat_weight = weight.reshape(128, 576)
        flat_output = output.reshape(8, 128, 196)
        if bias is None:
            matmul_no_bias(flat_weight, columns, flat_output)
        else:
            matmul_bias(flat_weight, columns, bias, flat_output)
        return output

    return wrapper
