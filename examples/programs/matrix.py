"""Contraction inputs, accumulators and storage formats are explicit."""

import torch

from kernels.contraction.batched_gemm import batched_gemm_nn
from kernels.contraction.block_scaled import block_scaled_matmul
from kernels.contraction.gemm import Activation, gemm as gemm_definition, gemm_i8
from kernels.quantization.quantized_projection import quantized_projection
from kernels.ragged.grouped_gemm import ragged_grouped_gemm


def gemm(context):
    kernel = context.compile(gemm_definition, constexprs={"ACTIVATION": Activation.NONE})
    a = torch.randn((128, 256), device=context.device, dtype=torch.float16)
    b = torch.randn((256, 192), device=context.device, dtype=torch.float16)
    return context.call(kernel, a, b)


def batched_gemm(context):
    kernel = context.compile(batched_gemm_nn)
    a = torch.randn((4, 128, 256), device=context.device, dtype=torch.bfloat16)
    b = torch.randn((4, 256, 192), device=context.device, dtype=torch.bfloat16)
    return context.call(kernel, a, b)


def ragged_gemm(context):
    kernel = context.compile(ragged_grouped_gemm)
    lengths = torch.tensor([17, 31, 48, 32], device=context.device, dtype=torch.int32)
    offsets = torch.cat((torch.zeros((1,), device=context.device, dtype=torch.int32),
                         lengths.cumsum(0).to(torch.int32)))
    x = torch.randn((128, 256), device=context.device, dtype=torch.float16)
    weight = torch.randn((4, 256, 192), device=context.device, dtype=torch.float16)
    return context.call(kernel, x, offsets, weight)


def int8_gemm(context):
    kernel = context.compile(gemm_i8)
    a = torch.randint(-16, 16, (128, 256), device=context.device, dtype=torch.int8)
    b = torch.randint(-16, 16, (256, 192), device=context.device, dtype=torch.int8)
    bias = torch.arange(192, device=context.device, dtype=torch.int32)
    return context.call(kernel, a, b, bias)


def q4_projection(context):
    kernel = context.compile(quantized_projection)
    # A zero-filled 144-byte Q4_K record has zero scales and represents zero.
    # Real callers supply their model's existing Q4_K records in this layout.
    weights = torch.zeros((128, 4, 144), device=context.device, dtype=torch.uint8)
    activation = torch.randn((4, 256), device=context.device, dtype=torch.float32)
    return context.call(kernel, weights, activation)


def fp8_gemm(context):
    kernel = context.compile(block_scaled_matmul)
    lhs = torch.randn((128, 8, 32), device=context.device).to(torch.float8_e4m3fn)
    rhs = torch.randn((8, 32, 128), device=context.device).to(torch.float8_e4m3fn)
    lhs_scale = torch.full((128, 8), 127, device=context.device, dtype=torch.uint8)
    rhs_scale = torch.full((8, 128), 127, device=context.device, dtype=torch.uint8)
    return context.call(kernel, lhs, lhs_scale, rhs, rhs_scale)
