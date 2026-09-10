import torch
import intent
import intent.language as I


@intent.kernel
def _gelu_conv2d(
    input: I.In[I.f32, ("B", "CI", "H", "W")],
    weight: I.In[I.f32, ("CO", "CI", "KH", "KW")],
    output: I.Out[I.f32, ("B", "CO", "H", "W")],
):
    batches = I.domain(0, input.shape[0])
    output_channels = I.domain(0, weight.shape[0])
    output_height = I.domain(0, input.shape[2])
    output_width = I.domain(0, input.shape[3])
    input_channels = I.domain(0, input.shape[1])
    kernel_height = I.domain(0, weight.shape[2])
    kernel_width = I.domain(0, weight.shape[3])

    for batch in I.parallel(batches):
        for channel_out in I.parallel(output_channels):
            for row in I.parallel(output_height):
                for column in I.parallel(output_width):
                    value = I.cast(0, I.f32)

                    for kernel_row in kernel_height:
                        input_row = (
                            I.cast(row, I.i64)
                            + I.cast(kernel_row, I.i64)
                            + I.cast(-1, I.i64)
                        )
                        if input_row >= 0:
                            if input_row < input.shape[2]:
                                for kernel_column in kernel_width:
                                    input_column = (
                                        I.cast(column, I.i64)
                                        + I.cast(kernel_column, I.i64)
                                        + I.cast(-1, I.i64)
                                    )
                                    if input_column >= 0:
                                        if input_column < input.shape[3]:
                                            input_vector = input[
                                                batch,
                                                input_channels,
                                                input_row,
                                                input_column,
                                            ]
                                            weight_vector = weight[
                                                channel_out,
                                                input_channels,
                                                kernel_row,
                                                kernel_column,
                                            ]
                                            value = value + I.dot(
                                                input_vector,
                                                weight_vector,
                                                acc_dtype=I.f32,
                                            )

                    scaled = value * I.cast(0.7071067811865475, I.f32)
                    if value < I.cast(0.0, I.f32):
                        absolute = scaled * I.cast(-1.0, I.f32)
                        sign = I.cast(-1.0, I.f32)
                    else:
                        absolute = scaled
                        sign = I.cast(1.0, I.f32)

                    t = I.fdiv(
                        I.cast(1.0, I.f32),
                        I.cast(1.0, I.f32)
                        + I.cast(0.3275911, I.f32) * absolute,
                    )
                    polynomial = (
                        I.cast(1.061405429, I.f32) * t
                        + I.cast(-1.453152027, I.f32)
                    )
                    polynomial = polynomial * t + I.cast(1.421413741, I.f32)
                    polynomial = polynomial * t + I.cast(-0.284496736, I.f32)
                    polynomial = polynomial * t + I.cast(0.254829592, I.f32)
                    polynomial = polynomial * t
                    exponent = absolute * absolute * I.cast(-1.4426950408889634, I.f32)
                    erf_absolute = I.cast(1.0, I.f32) + (
                        polynomial * I.exp2(exponent) * I.cast(-1.0, I.f32)
                    )
                    output[batch, channel_out, row, column] = (
                        I.cast(0.5, I.f32)
                        * value
                        * (I.cast(1.0, I.f32) + sign * erf_absolute)
                    )


def build(context):
    compiled = context.compile("gelu_conv2d", _gelu_conv2d)

    def wrapper(
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        approximate="none",
        out=None,
    ):
        output = out
        if output is None:
            output = torch.empty(
                (input.shape[0], weight.shape[0], input.shape[2], input.shape[3]),
                dtype=input.dtype,
                device=input.device,
            )
        compiled(input, weight, output)
        return output

    return wrapper
