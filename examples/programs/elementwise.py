"""Elementwise and mutable-state examples; target is selected by the host."""

import torch

from kernels.activation.pointwise import relu_forward
from kernels.activation.swiglu import swiglu_forward
from kernels.layout.transpose import matrix_transpose
from kernels.optimization.adamw import adamw_update
from kernels.position.rope import HEAD_DIMENSION, ROPE_QUERY_HEADS, ROPE_KEY_HEADS, rotary_qk_inplace
from kernels.regularization.dropout import xor_shift_dropout


def relu(context):
    kernel = context.compile(relu_forward)
    x = torch.randn((64, 1024), device=context.device, dtype=torch.float16)
    return context.call(kernel, x)


def swiglu(context):
    kernel = context.compile(swiglu_forward)
    gate = torch.randn((64, 1024), device=context.device, dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    return context.call(kernel, gate, up)


def rope(context):
    kernel = context.compile(rotary_qk_inplace)
    query = torch.randn((2, ROPE_QUERY_HEADS, 128, HEAD_DIMENSION),
                        device=context.device, dtype=torch.float16)
    key = torch.randn((2, ROPE_KEY_HEADS, 128, HEAD_DIMENSION),
                      device=context.device, dtype=torch.float16)
    phase = torch.randn((1, 128, HEAD_DIMENSION), device=context.device, dtype=torch.float16)
    context.call(kernel, query, key, phase.cos(), phase.sin())
    return query, key


def transpose(context):
    kernel = context.compile(matrix_transpose)
    x = torch.randn((127, 257), device=context.device, dtype=torch.float16)
    return context.call(kernel, x)


def dropout(context):
    kernel = context.compile(xor_shift_dropout)
    x = torch.randn((64, 1024), device=context.device, dtype=torch.float16)
    probability = 0.1
    return context.call(kernel, x, 12345, probability, 1.0 / (1.0 - probability))


def adamw(context):
    kernel = context.compile(adamw_update)
    gradient = torch.randn((65536,), device=context.device, dtype=torch.float32)
    parameter = torch.randn_like(gradient)
    first_moment = torch.zeros_like(gradient)
    second_moment = torch.zeros_like(gradient)
    beta1, beta2 = 0.9, 0.999
    context.call(kernel, gradient, parameter, first_moment, second_moment,
                 1.0e-3, beta1, beta2, 1.0 - beta1, 1.0 - beta2, 1.0e-8, 0.01)
    return parameter, first_moment, second_moment
