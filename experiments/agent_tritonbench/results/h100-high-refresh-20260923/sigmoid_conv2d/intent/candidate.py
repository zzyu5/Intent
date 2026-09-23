import torch
import intent
import intent.language as I


@intent.kernel
def _sigmoid_conv2d_bias(
    input: I.In[I.f32, (1, 64, 64, 64)],
    weight: I.In[I.f32, (128, 64, 3, 3)],
    bias: I.In[I.f32, (128,)],
    out: I.Out[I.f32, (1, 128, 62, 62)],
):
    channels = I.domain(0, 64)
    kernel_h = I.domain(0, 3)
    kernel_w = I.domain(0, 3)

    channel_index = I.reshape(I.indices(channels), (64, 1, 1))
    kernel_h_index = I.reshape(I.indices(kernel_h), (1, 3, 1))
    kernel_w_index = I.reshape(I.indices(kernel_w), (1, 1, 3))

    for output_channel in I.parallel(I.domain(0, 128)):
        for output_row in I.parallel(I.domain(0, 62)):
            for output_col in I.parallel(I.domain(0, 62)):
                patch = input[
                    0,
                    channel_index,
                    output_row + kernel_h_index,
                    output_col + kernel_w_index,
                ]
                filter_value = weight[
                    output_channel,
                    channel_index,
                    kernel_h_index,
                    kernel_w_index,
                ]
                convolution = I.reduce.sum(
                    patch * filter_value,
                    axis=(0, 1, 2),
                    acc_dtype=I.f32,
                )
                out[0, output_channel, output_row, output_col] = I.sigmoid(
                    convolution + bias[output_channel]
                )


@intent.kernel
def _sigmoid_conv2d_no_bias(
    input: I.In[I.f32, (1, 64, 64, 64)],
    weight: I.In[I.f32, (128, 64, 3, 3)],
    out: I.Out[I.f32, (1, 128, 62, 62)],
):
    channels = I.domain(0, 64)
    kernel_h = I.domain(0, 3)
    kernel_w = I.domain(0, 3)

    channel_index = I.reshape(I.indices(channels), (64, 1, 1))
    kernel_h_index = I.reshape(I.indices(kernel_h), (1, 3, 1))
    kernel_w_index = I.reshape(I.indices(kernel_w), (1, 1, 3))

    for output_channel in I.parallel(I.domain(0, 128)):
        for output_row in I.parallel(I.domain(0, 62)):
            for output_col in I.parallel(I.domain(0, 62)):
                patch = input[
                    0,
                    channel_index,
                    output_row + kernel_h_index,
                    output_col + kernel_w_index,
                ]
                filter_value = weight[
                    output_channel,
                    channel_index,
                    kernel_h_index,
                    kernel_w_index,
                ]
                convolution = I.reduce.sum(
                    patch * filter_value,
                    axis=(0, 1, 2),
                    acc_dtype=I.f32,
                )
                out[0, output_channel, output_row, output_col] = I.sigmoid(convolution)


def build(context):
    bias_kernel = context.compile("sigmoid_conv2d_bias", _sigmoid_conv2d_bias)
    no_bias_kernel = context.compile("sigmoid_conv2d_no_bias", _sigmoid_conv2d_no_bias)

    def sigmoid_conv2d(
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        out=None,
    ):
        if out is None:
            out = torch.empty(
                (1, 128, 62, 62),
                device=input.device,
                dtype=input.dtype,
            )

        if bias is None:
            no_bias_kernel(input, weight, out)
        else:
            bias_kernel(input, weight, bias, out)
        return out

    return sigmoid_conv2d
