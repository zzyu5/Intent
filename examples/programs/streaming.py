"""Stateful kernels and author-owned partial/merge orchestration."""

import torch

from kernels.streaming.attention import flash_attention_bf16_fwd
from kernels.streaming.gated_delta import recurrent_gated_delta_fwd
from kernels.streaming.mamba import mamba_chunk_state_bf16_fwd
from kernels.streaming.paged_attention import paged_gqa_decode_partials
from kernels.streaming.splitk_reduce import splitk_attention_f32_to_f16_reduce
from .composition import PagedDecode


def attention(context):
    kernel = context.compile(flash_attention_bf16_fwd,
                             constexprs={"HEAD_GROUP": 4, "CAUSAL": True})
    q = torch.randn((2, 8, 256, 64), device=context.device, dtype=torch.bfloat16)
    k = torch.randn((2, 2, 256, 64), device=context.device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    return context.call(kernel, q, k, v, 64**-0.5)


def paged_decode(context):
    batch, query_heads, kv_heads, dimension = 2, 8, 2, 64
    sequence, page_size, splits = 512, 16, 8
    pages_per_sequence = sequence // page_size
    pages = batch * pages_per_sequence
    partials = context.compile(paged_gqa_decode_partials, constexprs={
        "PAGE_SIZE": page_size, "HEAD_GROUP": query_heads // kv_heads, "SPLITS": splits})
    merge = context.compile(splitk_attention_f32_to_f16_reduce)
    q = torch.randn((batch, query_heads, dimension), device=context.device, dtype=torch.float16)
    key_cache = torch.randn((pages, page_size, kv_heads, dimension),
                            device=context.device, dtype=torch.float16)
    value_cache = torch.randn_like(key_cache)
    page_indices = torch.arange(pages, device=context.device, dtype=torch.int32)
    page_offsets = torch.arange(0, pages + 1, pages_per_sequence,
                                device=context.device, dtype=torch.int32)
    split_offsets = torch.arange(0, pages + 1, pages_per_sequence // splits,
                                 device=context.device, dtype=torch.int32)
    lengths = torch.full((batch,), sequence, device=context.device, dtype=torch.int32)
    program = PagedDecode(partials, merge)
    return context.call(program, q, key_cache, value_cache, page_offsets, page_indices,
                        lengths, split_offsets, dimension**-0.5)


def mamba_chunk_state(context):
    kernel = context.compile(mamba_chunk_state_bf16_fwd, constexprs={"HEAD_GROUP": 4})
    batch, chunks, chunk_size, heads, groups = 2, 4, 64, 8, 2
    dimension, state_dimension = 64, 32
    basis = torch.randn((batch, chunks * chunk_size, groups, state_dimension),
                        device=context.device, dtype=torch.bfloat16)
    x = torch.randn((batch, chunks * chunk_size, heads, dimension),
                    device=context.device, dtype=torch.bfloat16)
    dt = torch.rand((batch, heads, chunks, chunk_size), device=context.device, dtype=torch.float32)
    cumulative_decay = (-torch.rand_like(dt) * 0.01).cumsum(-1)
    return context.call(kernel, basis, x, dt, cumulative_decay)


def gated_delta(context):
    kernel = context.compile(recurrent_gated_delta_fwd, constexprs={"HEAD_GROUP": 2})
    q = torch.randn((2, 128, 2, 32), device=context.device, dtype=torch.bfloat16) * 0.05
    k = torch.randn_like(q) * 0.05
    v = torch.randn((2, 128, 4, 32), device=context.device, dtype=torch.bfloat16)
    gate = (-torch.rand((2, 128, 4), device=context.device) * 0.01).to(torch.bfloat16)
    beta = torch.rand((2, 128, 4), device=context.device, dtype=torch.bfloat16)
    return context.call(kernel, q, k, v, gate, beta, 32**-0.5)
