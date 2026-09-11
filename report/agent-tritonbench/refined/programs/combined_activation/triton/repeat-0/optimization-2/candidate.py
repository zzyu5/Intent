import torch
import triton
import triton.language as tl


@triton.jit
def _combined_activation_kernel(
    input_ptr,
    weight1_ptr,
    weight2_ptr,
    bias_ptr,
    output_ptr,
    m,
    n,
    k,
    input_stride_m,
    input_stride_k,
    weight1_stride_k,
    weight1_stride_n,
    weight2_stride_n,
    bias_stride_n,
    output_stride_m,
    output_stride_n,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * 64 + tl.arange(0, 64)
    cols = pid_n * 64 + tl.arange(0, 64)

    acc = tl.zeros((64, 64), dtype=tl.float32)
    for k_start in range(0, 512, 64):
        k_offsets = k_start + tl.arange(0, 64)

        input_offsets = (
            input_ptr
            + rows[:, None] * input_stride_m
            + k_offsets[None, :] * input_stride_k
        )
        weight1_offsets = (
            weight1_ptr
            + k_offsets[:, None] * weight1_stride_k
            + cols[None, :] * weight1_stride_n
        )
        lhs = tl.load(input_offsets)
        rhs = tl.load(weight1_offsets)
        acc = tl.dot(lhs, rhs, acc=acc, input_precision="ieee")

    sigmoid = 1.0 / (1.0 + tl.exp(-acc))
    exp_two_sigmoid = tl.exp(2.0 * sigmoid)
    activation = 1.0 - 2.0 / (exp_two_sigmoid + 1.0)

    weight2 = tl.load(
        weight2_ptr + cols * weight2_stride_n,
    )
    bias = tl.load(
        bias_ptr + cols * bias_stride_n,
    )
    result = activation * weight2[None, :] + bias[None, :]

    output_offsets = (
        output_ptr
        + rows[:, None] * output_stride_m
        + cols[None, :] * output_stride_n
    )
    tl.store(output_offsets, result)


def build(context):
    def combined_activation(input, weight1, weight2, bias, *, out=None):
        k = input.shape[-1]
        n = weight1.shape[-1]
        m = input.numel() // k

        if out is None:
            output_shape = input.shape[:-1] + (n,)
            output = torch.empty(output_shape, device=input.device, dtype=input.dtype)
        else:
            output = out

        _combined_activation_kernel[(8, 8)](
            input,
            weight1,
            weight2,
            bias,
            output,
            m,
            n,
            k,
            input.stride(-2),
            input.stride(-1),
            weight1.stride(-2),
            weight1.stride(-1),
            weight2.stride(-1),
            bias.stride(-1),
            output.stride(-2),
            output.stride(-1),
        )
        return output

    return combined_activation
