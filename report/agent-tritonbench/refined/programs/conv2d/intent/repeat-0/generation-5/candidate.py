import torch
import intent
import intent.language as I


@intent.kernel
def conv2d_im2col(
    x: I.In[I.f32, (8, 64, 16, 16)],
    columns: I.Out[I.f32, (8, 64, 3, 3, 14, 14)],
):
    input_rows = I.domain(0, 16)
    input_cols = I.domain(0, 16)
    output_rows = I.domain(0, 14)
    output_cols = I.domain(0, 14)

    for batch in I.parallel(I.domain(0, 8)):
        for channel in I.parallel(I.domain(0, 64)):
            for kernel_row in I.parallel(I.domain(0, 3)):
                rows = input_rows[kernel_row : kernel_row + 14]
                for kernel_col in I.parallel(I.domain(0, 3)):
                    cols = input_cols[kernel_col : kernel_col + 14]
                    columns[batch, channel, kernel_row, kernel_col, output_rows, output_cols] = x[
                        batch, channel, rows, cols
                    ]


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
        columns_storage = torch.empty((8, 64, 3, 3, 14, 14), device=input.device, dtype=input.dtype)

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
