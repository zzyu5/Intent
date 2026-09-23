import torch
import intent
import intent.language as I


@intent.kernel
def _sigmoid_adaptive_avg_pool2d_kernel(
    input: I.In[I.f32, (2, 2, 16, 16)],
    output: I.Out[I.f32, (2, 2, 1, 1)],
):
    batches = I.domain(0, 2)
    channels = I.domain(0, 2)
    input_rows = I.domain(0, 16)
    input_cols = I.domain(0, 16)
    output_rows = I.domain(0, 1)
    output_cols = I.domain(0, 1)

    spatial_values = input[batches, channels, input_rows, input_cols]
    spatial_sum = I.reduce.sum(spatial_values, axis=(2, 3), acc_dtype=I.f32)
    spatial_mean = I.fdiv(spatial_sum, I.cast(256.0, I.f32))
    activated = I.sigmoid(spatial_mean)
    output[batches, channels, output_rows, output_cols] = I.reshape(
        activated, (2, 2, 1, 1)
    )


def build(context):
    kernel = context.compile("sigmoid_adaptive_avg_pool2d", _sigmoid_adaptive_avg_pool2d_kernel)

    def sigmoid_adaptive_avg_pool2d(input: torch.Tensor, output_size):
        if isinstance(output_size, int):
            output_shape = (output_size, output_size)
        else:
            output_shape = tuple(output_size)
        if output_shape != (1, 1):
            raise ValueError("the supplied profile requires output_size=(1, 1)")

        output = torch.empty(
            (input.shape[0], input.shape[1], 1, 1),
            device=input.device,
            dtype=input.dtype,
        )
        kernel(input, output)
        return output

    return sigmoid_adaptive_avg_pool2d
