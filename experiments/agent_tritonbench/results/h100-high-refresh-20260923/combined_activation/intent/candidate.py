import torch
import intent
import intent.language as I


@intent.kernel
def _combined_activation_kernel(
    input: I.In[I.f32, ("M", "K")],
    weight1: I.In[I.f32, ("K", "N")],
    weight2: I.In[I.f32, ("N",)],
    bias: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("M", "N")],
):
    M, K = input.shape
    N = weight1.shape[1]

    rows = I.domain(0, M)
    columns = I.domain(0, N)
    reduction = I.domain(0, K)

    # Form [M, K, N] products and reduce the K axis as the GEMM stage.
    lhs = I.reshape(input[rows, reduction], (M, 1, K))
    rhs = I.reshape(weight1[reduction, columns], (1, K, N))
    matmul = I.reduce.sum(lhs * rhs, axis=1, acc_dtype=I.f32)

    one = I.cast(1.0, I.f32)
    sigmoid = I.fdiv(one, one + I.exp(-matmul))
    activated = I.tanh(sigmoid)
    output[rows, columns] = activated * weight2[columns] + bias[columns]


def build(context):
    kernel = context.compile("combined_activation_fused", _combined_activation_kernel)

    def combined_activation(input, weight1, weight2, bias, *, out=None):
        result = out
        if result is None:
            result = torch.empty(
                (input.shape[0], weight1.shape[1]),
                device=input.device,
                dtype=input.dtype,
            )
        kernel(input, weight1, weight2, bias, result)
        return result

    return combined_activation
