import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _combined_activation_kernel(
    input_ptr,
    weight1_ptr,
    weight2_ptr,
    bias_ptr,
    output_ptr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * 64 + tl.arange(0, 64)
    cols = pid_n * 64 + tl.arange(0, 64)

    acc = tl.zeros((64, 64), dtype=tl.float32)
    for k_start in range(0, 512, 64):
        k_offsets = k_start + tl.arange(0, 64)
        lhs = tl.load(input_ptr + rows[:, None] * 512 + k_offsets[None, :])
        rhs = tl.load(weight1_ptr + k_offsets[:, None] * 512 + cols[None, :])
        acc = tl.dot(lhs, rhs, acc=acc, input_precision="ieee")

    sigmoid = 1.0 / (1.0 + libdevice.fast_expf(-acc))
    exp_two_sigmoid = libdevice.fast_expf(2.0 * sigmoid)
    activation = 1.0 - 2.0 / (exp_two_sigmoid + 1.0)
    weight2 = tl.load(weight2_ptr + cols)
    bias = tl.load(bias_ptr + cols)
    result = activation * weight2[None, :] + bias[None, :]

    tl.store(output_ptr + rows[:, None] * 512 + cols[None, :], result)


def build(context):
    def combined_activation(input, weight1, weight2, bias, *, out=None):
        if out is None:
            output = torch.empty_like(input)
        else:
            output = out

        _combined_activation_kernel[(8, 8)](input, weight1, weight2, bias, output)
        return output

    return combined_activation
