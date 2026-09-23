import torch
import intent
import intent.language as I


@intent.fn
def _selu(value):
    alpha = I.cast(1.6732632423543772, I.f32)
    scale = I.cast(1.0507009873554805, I.f32)
    zero = I.cast(0.0, I.f32)
    one = I.cast(1.0, I.f32)
    return I.select(
        value > zero,
        scale * value,
        scale * alpha * (I.exp(value) - one),
    )


@intent.kernel
def _conv_selu(
    input: I.In[I.f32, (2, 16, 64, 64)],
    weight: I.In[I.f32, (64, 16, 3, 3)],
    bias: I.In[I.f32, (64,)],
    output: I.Out[I.f32, (2, 64, 62, 62)],
):
    rows = I.domain(0, 62)
    cols = I.domain(0, 62)
    row_index = I.reshape(I.indices(rows), (62, 1))
    col_index = I.reshape(I.indices(cols), (1, 62))

    for batch in I.parallel(I.domain(0, 2)):
        for channel in I.parallel(I.domain(0, 64)):
            value = I.full((62, 62), 0.0, dtype=I.f32)
            for input_channel in I.domain(0, 16):
                for kernel_row in I.domain(0, 3):
                    for kernel_col in I.domain(0, 3):
                        source = input[
                            batch,
                            input_channel,
                            row_index + kernel_row,
                            col_index + kernel_col,
                        ]
                        coefficient = weight[
                            channel,
                            input_channel,
                            kernel_row,
                            kernel_col,
                        ]
                        value = value + source * coefficient
            value = value + bias[channel]
            output[batch, channel, rows, cols] = _selu(value)


@intent.kernel
def _instance_norm(
    input: I.In[I.f32, (2, 64, 62, 62)],
    output: I.Out[I.f32, (2, 64, 62, 62)],
    eps: I.f32,
):
    rows = I.domain(0, 62)
    cols = I.domain(0, 62)
    count = I.cast(3844.0, I.f32)

    for batch in I.parallel(I.domain(0, 2)):
        for channel in I.parallel(I.domain(0, 64)):
            plane = input[batch, channel, rows, cols]
            total = I.reduce.sum(plane, axis=(0, 1), acc_dtype=I.f32)
            mean = total / count
            centered = plane - mean
            squared = centered * centered
            variance = I.reduce.sum(squared, axis=(0, 1), acc_dtype=I.f32) / count
            output[batch, channel, rows, cols] = centered / I.sqrt(variance + eps)


def build(context):
    conv_selu = context.compile("conv_selu", _conv_selu)
    instance_norm = context.compile("instance_norm", _instance_norm)

    def fused_instance_norm_selu_conv2d(
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        num_features=None,
        eps=1e-5,
        momentum=0.1,
        affine=False,
        track_running_stats=False,
    ):
        output_height = input.shape[2] - 2
        output_width = input.shape[3] - 2
        intermediate = torch.empty(
            (input.shape[0], weight.shape[0], output_height, output_width),
            device=input.device,
            dtype=input.dtype,
        )
        output = torch.empty_like(intermediate)
        conv_selu(input, weight, bias, intermediate)
        instance_norm(intermediate, output, eps)
        return output

    return fused_instance_norm_selu_conv2d
