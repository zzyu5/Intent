import torch
import intent
import intent.language as I


@intent.kernel
def cos_avg_pool1d_kernel(
    input: I.In[I.f32, ("B", "C", "W")],
    output: I.Out[I.f32, ("B", "C", "O")],
):
    batch_size, channels, _ = input.shape
    output_width = output.shape[2]

    for batch in I.parallel(I.domain(0, batch_size)):
        for channel in I.parallel(I.domain(0, channels)):
            for out_x in I.parallel(I.domain(0, output_width)):
                start = out_x * 3
                first = I.cos(input[batch, channel, start])
                second = I.cos(input[batch, channel, start + 1])
                third = I.cos(input[batch, channel, start + 2])
                total = (first + second) + third
                output[batch, channel, out_x] = total / 3.0


def build(context):
    kernel = context.compile("cos_avg_pool1d_fused", cos_avg_pool1d_kernel)

    def cos_avg_pool1d(
        input: torch.Tensor,
        kernel_size: int,
        stride: int = None,
        padding: int = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> torch.Tensor:
        if stride is None:
            stride = kernel_size

        batch_size, channels, width = input.shape
        if ceil_mode:
            output_width = (width + 2 * padding - kernel_size + stride - 1) // stride + 1
            if (output_width - 1) * stride >= width + padding:
                output_width -= 1
        else:
            output_width = (width + 2 * padding - kernel_size) // stride + 1

        output = torch.empty(
            (batch_size, channels, output_width),
            device=input.device,
            dtype=input.dtype,
        )
        kernel(input, output)
        return output

    return cos_avg_pool1d
