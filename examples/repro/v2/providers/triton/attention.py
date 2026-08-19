from __future__ import annotations

import math

import torch

from kernels.backward.attention import attention_backward_delta
from kernels.backward.attention import attention_backward_dkdv
from kernels.backward.attention import attention_backward_dq
from kernels.streaming.attention import flash_attention_fwd
from kernels.streaming.block_sparse_attention import block_sparse_gqa_decode_combine
from kernels.streaming.block_sparse_attention import block_sparse_gqa_decode_partials
from kernels.streaming.mla import paged_mla_decode
from kernels.streaming.paged_attention import paged_gqa_decode_attention

from ...loading import load_module
from ...measurement import compile_single
from ...measurement import functional_launch
from ...model import Context
from ...model import PreparedComparison
from ...model import PreparedLaunch
from ...model import Tolerance


def flash_attention_forward(context: Context) -> PreparedComparison:
    batch, heads, sequence, dimension = 4, 32, 4096, 128
    shape = (batch, heads, sequence, dimension)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn(shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, device="cuda", dtype=torch.float16)
    scale = 1.0 / math.sqrt(dimension)
    _, generated = compile_single(
        context,
        flash_attention_fwd,
        (q, k, v, scale),
        constexprs={"CAUSAL": True},
    )
    runtime = load_module(
        context.project_root
        / "source/triton/triton/attention/fused/06-fused-attention_runtime.py",
        "intent_v2_triton_flash_attention",
    )
    source_function = runtime.load_attention()
    warp_specialize = torch.cuda.get_device_capability()[0] >= 10
    source = functional_launch(
        lambda: source_function(q, k, v, True, scale, warp_specialize)
    )
    return PreparedComparison(generated, source, Tolerance(atol=2e-2, rtol=2e-2), cuda_graph=True)


def paged_gqa_decode(context: Context) -> PreparedComparison:
    batch, query_heads, kv_heads, dimension = 16, 32, 8, 128
    sequence, page_size, splits = 8192, 16, 8
    pages_per_sequence = sequence // page_size
    pages = batch * pages_per_sequence
    q = torch.randn(
        (batch, query_heads, dimension), device="cuda", dtype=torch.float16
    )
    key_cache = torch.randn(
        (pages, page_size, kv_heads, dimension),
        device="cuda",
        dtype=torch.float16,
    )
    value_cache = torch.randn_like(key_cache)
    page_indices = torch.arange(pages, device="cuda", dtype=torch.int32)
    page_offsets = torch.arange(
        0,
        pages + 1,
        pages_per_sequence,
        device="cuda",
        dtype=torch.int32,
    )
    page_table = page_indices.view(batch, pages_per_sequence)
    lengths = torch.full(
        (batch,), sequence, device="cuda", dtype=torch.int32
    )
    scale = dimension**-0.5
    _, generated = compile_single(
        context,
        paged_gqa_decode_attention,
        (
            q,
            key_cache,
            value_cache,
            page_offsets,
            page_indices,
            lengths,
            scale,
        ),
        constexprs={"PAGE_SIZE": page_size, "HEAD_GROUP": query_heads // kv_heads},
    )
    runtime = load_module(
        context.project_root
        / "source/triton/vllm/attention/paged_decode/paged_gqa_decode_runtime.py",
        "intent_v2_triton_paged_gqa_runtime",
    )
    source_module = runtime.RUNTIME.load_source(
        context.project_root
        / "source/triton/vllm/attention/paged_decode/triton_decode_attention.py",
        "intent_v2_triton_paged_gqa_source",
    )
    source_output = torch.empty_like(q)
    source_lse = torch.empty(
        (batch, query_heads), device="cuda", dtype=torch.float32
    )
    workspace = torch.empty(
        (batch, query_heads, splits, dimension + 1),
        device="cuda",
        dtype=torch.float32,
    )

    def launch():
        source_module.decode_attention_fwd(
            q,
            key_cache,
            value_cache,
            source_output,
            source_lse,
            page_table,
            lengths,
            workspace,
            splits,
            scale,
            page_size=page_size,
        )

    launch()
    source = PreparedLaunch(launch=launch, outputs=lambda: source_output)
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=5e-2, rtol=5e-2),
        cuda_graph=False,
    )


def paged_mla(context: Context) -> PreparedComparison:
    batch, query_heads, kv_heads = 8, 128, 1
    sequence, latent_dimension, rope_dimension = 8192, 512, 64
    key_dimension = latent_dimension + rope_dimension
    page_size, splits = 16, 8
    pages_per_sequence = sequence // page_size
    pages = batch * pages_per_sequence
    q_latent = torch.randn(
        (batch, query_heads, latent_dimension),
        device="cuda",
        dtype=torch.float16,
    )
    q_rope = torch.randn(
        (batch, query_heads, rope_dimension),
        device="cuda",
        dtype=torch.float16,
    )
    latent_cache = torch.randn(
        (pages, page_size, kv_heads, latent_dimension),
        device="cuda",
        dtype=torch.float16,
    )
    rope_cache = torch.randn(
        (pages, page_size, kv_heads, rope_dimension),
        device="cuda",
        dtype=torch.float16,
    )
    page_indices = torch.arange(pages, device="cuda", dtype=torch.int32)
    page_offsets = torch.arange(
        0,
        pages + 1,
        pages_per_sequence,
        device="cuda",
        dtype=torch.int32,
    )
    page_table = page_indices.view(batch, pages_per_sequence)
    lengths = torch.full(
        (batch,), sequence, device="cuda", dtype=torch.int32
    )
    scale = key_dimension**-0.5
    _, generated = compile_single(
        context,
        paged_mla_decode,
        (
            q_latent,
            q_rope,
            latent_cache,
            rope_cache,
            page_offsets,
            page_indices,
            lengths,
            scale,
        ),
        constexprs={"PAGE_SIZE": page_size, "HEAD_GROUP": query_heads // kv_heads},
    )
    runtime = load_module(
        context.project_root
        / "source/triton/vllm/attention/paged_decode/paged_mla_decode_runtime.py",
        "intent_v2_triton_paged_mla_runtime",
    )
    source_module = runtime.RUNTIME.load_source(
        context.project_root
        / "source/triton/vllm/attention/paged_decode/triton_decode_attention.py",
        "intent_v2_triton_paged_mla_source",
    )
    q = torch.cat((q_latent, q_rope), dim=-1)
    key = torch.cat((latent_cache, rope_cache), dim=-1)
    source_output = torch.empty(
        (batch, query_heads, latent_dimension),
        device="cuda",
        dtype=torch.float16,
    )
    source_lse = torch.empty(
        (batch, query_heads), device="cuda", dtype=torch.float32
    )
    workspace = torch.empty(
        (batch, query_heads, splits, latent_dimension + 1),
        device="cuda",
        dtype=torch.float32,
    )

    def launch():
        source_module.decode_attention_fwd(
            q,
            key,
            latent_cache,
            source_output,
            source_lse,
            page_table,
            lengths,
            workspace,
            splits,
            scale,
            page_size=page_size,
            is_mla=True,
        )

    launch()
    source = PreparedLaunch(launch=launch, outputs=lambda: source_output)
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=5e-2, rtol=5e-2),
        cuda_graph=False,
    )


def block_sparse_gqa_decode(context: Context) -> PreparedComparison:
    batch, query_heads, kv_heads, dimension = 8, 32, 8, 128
    sequence, selected_blocks, block_size, splits = 8192, 32, 128, 4
    blocks_per_sequence = sequence // block_size
    q = torch.randn(
        (batch, query_heads, dimension), device="cuda", dtype=torch.float16
    )
    key = torch.randn(
        (batch, sequence, kv_heads, dimension),
        device="cuda",
        dtype=torch.float16,
    )
    value = torch.randn_like(key)
    selected = torch.arange(
        selected_blocks, device="cuda", dtype=torch.int32
    ).view(1, 1, selected_blocks).expand(batch, kv_heads, selected_blocks).contiguous()
    lengths = torch.full(
        (batch,), sequence, device="cuda", dtype=torch.int32
    )
    split_offsets = torch.arange(
        0,
        selected_blocks + 1,
        selected_blocks // splits,
        device="cuda",
        dtype=torch.int32,
    )
    scale = dimension**-0.5
    _, partial = compile_single(
        context,
        block_sparse_gqa_decode_partials,
        (q, key, value, selected, lengths, split_offsets, scale),
        constexprs={
            "HEAD_GROUP": query_heads // kv_heads,
            "BLOCK_SIZE": block_size,
            "SPLITS": splits,
        },
    )
    partial_lse, partial_output = partial.outputs()
    _, combined = compile_single(
        context,
        block_sparse_gqa_decode_combine,
        (partial_lse, partial_output),
        constexprs={"SPLITS": splits},
    )

    def generated_launch():
        partial.launch()
        combined.launch()

    generated = PreparedLaunch(
        launch=generated_launch,
        outputs=combined.outputs,
    )

    runtime = load_module(
        context.project_root
        / "source/triton/vllm/attention/minimax_m3/sparse_decode_runtime.py",
        "intent_v2_triton_sparse_decode_runtime",
    )
    source_module = runtime.RUNTIME.load_source(
        context.project_root
        / "source/triton/vllm/attention/minimax_m3/sparse_attn.py",
        "intent_v2_triton_sparse_decode_source",
    )
    key_blocks = key.view(
        batch, blocks_per_sequence, block_size, kv_heads, dimension
    ).permute(0, 1, 3, 2, 4)
    value_blocks = value.view(
        batch, blocks_per_sequence, block_size, kv_heads, dimension
    ).permute(0, 1, 3, 2, 4)
    kv_cache = torch.cat((key_blocks, value_blocks), dim=-1).reshape(
        batch * blocks_per_sequence,
        kv_heads,
        block_size,
        2 * dimension,
    )
    block_table = torch.arange(
        batch * blocks_per_sequence, device="cuda", dtype=torch.int32
    ).view(batch, blocks_per_sequence)
    topk = selected.permute(1, 0, 2).contiguous()
    source_output = torch.empty_like(q)

    def source_launch():
        source_module.minimax_m3_sparse_attn_decode(
            q,
            kv_cache,
            topk,
            block_table,
            lengths,
            kv_heads,
            scale,
            source_output,
            decode_query_len=1,
        )

    source_launch()
    source = PreparedLaunch(launch=source_launch, outputs=lambda: source_output)
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=5e-2, rtol=5e-2),
        cuda_graph=False,
    )


def flash_attention_backward(context: Context) -> PreparedComparison:
    batch, heads, sequence, dimension = 2, 16, 2048, 128
    shape = (batch, heads, sequence, dimension)
    q = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.5
    k = torch.randn_like(q) * 0.5
    v = torch.randn_like(q) * 0.5
    output = torch.empty_like(q)
    lse = torch.empty(
        (batch, heads, sequence), device="cuda", dtype=torch.float32
    )
    runtime_helper = load_module(
        context.project_root / "source/triton/meta-applied-ai/support/runtime.py",
        "intent_v2_triton_flash_backward_runtime",
    )
    source_module = runtime_helper.load_source(
        context.project_root
        / "source/triton/meta-applied-ai/attention/flash_backward/flash_backward.py",
        "intent_v2_triton_flash_backward_source",
    )
    source_module.flash(q, k, v, output, lse)
    grad_output = torch.randn_like(output) * 0.05
    scale = dimension**-0.5
    _, delta = compile_single(
        context,
        attention_backward_delta,
        (output, grad_output),
    )
    delta_value = delta.outputs()
    generated_lse = lse / math.log2(math.e)
    common = (q, k, v, grad_output, generated_lse, delta_value, scale)
    constexprs = {"HEAD_GROUP": 1, "CAUSAL": True}
    _, dkdv = compile_single(
        context,
        attention_backward_dkdv,
        common,
        constexprs=constexprs,
    )
    _, dq = compile_single(
        context,
        attention_backward_dq,
        common,
        constexprs=constexprs,
    )

    def generated_launch():
        delta.launch()
        dkdv.launch()
        dq.launch()

    generated = PreparedLaunch(
        launch=generated_launch,
        outputs=lambda: (dq.outputs(), *dkdv.outputs()),
    )
    source = functional_launch(
        lambda: source_module.flash_bwd(q, k, v, output, lse, grad_output)
    )
    return PreparedComparison(
        generated,
        source,
        (
            Tolerance(atol=1.25e-1),
            Tolerance(atol=2.5e-1),
            Tolerance(atol=1.25e-1),
        ),
        cuda_graph=False,
    )


CASES = {
    "flash_attention_forward": flash_attention_forward,
    "paged_gqa_decode": paged_gqa_decode,
    "paged_mla_decode": paged_mla,
    "block_sparse_gqa_decode": block_sparse_gqa_decode,
    "flash_attention_backward": flash_attention_backward,
}
