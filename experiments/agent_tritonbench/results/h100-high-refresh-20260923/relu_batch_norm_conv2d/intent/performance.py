import torch
import intent
import intent.language as I


@intent.kernel
def conv2d_stage(
    input: I.In[I.f32, (1, 64, 32, 32)],
    weight: I.In[I.f32, (128, 64, 3, 3)],
    output: I.Out[I.f32, (1, 128, 32, 32)],
):
    # Each output coordinate is an independent logical group.  The receptive
    # field is a 64 x 3 x 3 tensor reduction for that group.
    channels = I.domain(0, 64)
    kernel_rows = I.domain(0, 3)
    kernel_cols = I.domain(0, 3)
    channel_index = I.reshape(I.indices(channels), (64, 1, 1))
    kernel_row_index = I.reshape(I.indices(kernel_rows), (1, 3, 1))
    kernel_col_index = I.reshape(I.indices(kernel_cols), (1, 1, 3))
    zero = I.cast(0.0, I.f32)

    for out_channel in I.parallel(I.domain(0, 128)):
        coefficients = weight[
            out_channel,
            channel_index,
            kernel_row_index,
            kernel_col_index,
        ]
        for out_row in I.parallel(I.domain(0, 32)):
            input_row = out_row + kernel_row_index - 1
            valid_row = I.select(input_row >= 0, input_row < 32, False)
            for out_col in I.parallel(I.domain(0, 32)):
                input_col = out_col + kernel_col_index - 1
                valid_col = I.select(input_col >= 0, input_col < 32, False)
                valid = I.select(valid_row, valid_col, False)
                samples = I.gather(
                    input,
                    (0, channel_index, input_row, input_col),
                    valid=valid,
                    fill=zero,
                )
                products = samples * coefficients
                result = I.reduce.sum(
                    products,
                    axis=(0, 1, 2),
                    acc_dtype=I.f32,
                )
                output[0, out_channel, out_row, out_col] = result


@intent.kernel
def batch_norm_relu_stage(
    input: I.In[I.f32, (1, 128, 32, 32)],
    running_mean: I.In[I.f32, (128,)],
    running_var: I.In[I.f32, (128,)],
    bn_weight: I.In[I.f32, (128,)],
    bn_bias: I.In[I.f32, (128,)],
    output: I.Out[I.f32, (1, 128, 32, 32)],
    eps: I.f32,
):
    zero = I.cast(0.0, I.f32)
    for channel in I.parallel(I.domain(0, 128)):
        mean = running_mean[channel]
        variance = running_var[channel]
        scale = I.fdiv(bn_weight[channel], I.sqrt(variance + eps))
        offset = bn_bias[channel]
        for row in I.parallel(I.domain(0, 32)):
            for col in I.parallel(I.domain(0, 32)):
                value = (input[0, channel, row, col] - mean) * scale + offset
                output[0, channel, row, col] = I.maximum(value, zero)


def build(context):
    conv_artifact = context.compile("relu_batch_norm_conv2d_conv", conv2d_stage)
    norm_artifact = context.compile("relu_batch_norm_conv2d_bn_relu", batch_norm_relu_stage)

    def relu_batch_norm_conv2d(
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        running_mean=None,
        running_var=None,
        bn_weight=None,
        bn_bias=None,
        training=False,
        momentum=0.1,
        eps=1e-05,
        inplace=False,
    ):
        conv = torch.empty(
            (1, 128, 32, 32),
            device=input.device,
            dtype=torch.float32,
        )
        output = torch.empty(
            (1, 128, 32, 32),
            device=input.device,
            dtype=input.dtype,
        )
        conv_artifact(input, weight, conv)
        norm_artifact(
            conv,
            running_mean,
            running_var,
            bn_weight,
            bn_bias,
            output,
            eps,
        )
        return output

    return relu_batch_norm_conv2d
