"""Statistics, prefixes and structured state use the original algorithms."""

import torch

from kernels.backward.group_norm import (
    BATCH, CHANNELS, SPATIAL, GROUPS, CHANNELS_PER_GROUP,
    group_norm_backward_dx, group_norm_backward_weight_bias,
)
from kernels.normalization.batch_norm import batch_norm_training
from kernels.normalization.layer_norm import weighted_layer_norm
from kernels.normalization.softmax import stable_softmax_f16
from kernels.statistics.histogram import histogram_256
from kernels.streaming.attention_f32 import causal_linear_attention_f32
from kernels.streaming.ordered_prefix import row_cumsum_f32
from .composition import GroupNormBackward


def softmax(context):
    kernel = context.compile(stable_softmax_f16)
    x = torch.randn((64, 1024), device=context.device, dtype=torch.float16)
    return context.call(kernel, x)


def layer_norm(context):
    kernel = context.compile(weighted_layer_norm)
    x = torch.randn((64, 1024), device=context.device, dtype=torch.float32)
    weight = torch.randn((1024,), device=context.device, dtype=torch.float32)
    bias = torch.randn_like(weight)
    return context.call(kernel, x, weight, bias, 1.0 / x.shape[1], 1.0e-5)


def batch_norm(context):
    kernel = context.compile(batch_norm_training)
    x = torch.randn((4, 16, 128), device=context.device, dtype=torch.float16)
    weight = torch.ones((16,), device=context.device, dtype=torch.float32)
    bias = torch.zeros_like(weight)
    running_mean = torch.zeros_like(weight)
    running_variance = torch.ones_like(weight)
    output, saved_mean, saved_rstd = context.call(
        kernel, x, weight, bias, running_mean, running_variance, 1.0e-5, 0.1)
    return output, saved_mean, saved_rstd, running_mean, running_variance


def group_norm_backward(context):
    dx = context.compile(group_norm_backward_dx)
    affine = context.compile(group_norm_backward_weight_bias)
    x = torch.randn((BATCH, CHANNELS, SPATIAL), device=context.device,
                    dtype=torch.float16) * 0.5
    grad_y = torch.randn_like(x) * 0.05
    weight = torch.randn((CHANNELS,), device=context.device, dtype=torch.float16)
    grouped = x.float().reshape(BATCH, GROUPS, CHANNELS_PER_GROUP, SPATIAL)
    mean = grouped.mean(dim=(2, 3)).to(torch.float16)
    centered = grouped - mean.float()[:, :, None, None]
    rstd = torch.rsqrt(centered.square().mean(dim=(2, 3)) + 1.0e-5).to(torch.float16)
    program = GroupNormBackward(dx, affine)
    return context.call(program, x, grad_y, weight, mean, rstd,
                        1.0 / (CHANNELS_PER_GROUP * SPATIAL))


def cumsum(context):
    kernel = context.compile(row_cumsum_f32)
    x = torch.randn((64, 1024), device=context.device, dtype=torch.float32)
    return context.call(kernel, x)


def causal_linear_attention(context):
    kernel = context.compile(causal_linear_attention_f32)
    q = torch.randn((2, 256, 32), device=context.device, dtype=torch.float32) * 0.1
    k = torch.randn_like(q) * 0.1
    v = torch.randn((2, 256, 64), device=context.device, dtype=torch.float32)
    return context.call(kernel, q, k, v)


def histogram(context):
    kernel = context.compile(histogram_256)
    samples = torch.randint(-16, 272, (65536,), device=context.device).float()
    return context.call(kernel, samples)
