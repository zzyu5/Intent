"""Host input preparation for ordered and neighborhood computations."""

import torch

from kernels.convolution.direct import causal_depthwise_conv1d
from kernels.factorization.cholesky import SIZE, batched_cholesky_lower


def causal_convolution(context):
    kernel = context.compile(causal_depthwise_conv1d, constexprs={"SILU": True})
    x = torch.randn((2, 16, 256), device=context.device, dtype=torch.float16)
    weight = torch.randn((16, 4), device=context.device, dtype=torch.float16)
    bias = torch.randn((16,), device=context.device, dtype=torch.float16)
    return context.call(kernel, x, weight, bias)


def cholesky(context):
    kernel = context.compile(batched_cholesky_lower)
    basis = torch.randn((16, SIZE, SIZE), device=context.device, dtype=torch.float32)
    matrices = basis @ basis.transpose(-1, -2) + torch.eye(
        SIZE, device=context.device, dtype=torch.float32)
    context.call(kernel, matrices)
    return matrices
