from __future__ import annotations

import itertools
import math

import torch

from kernels.backward.attention import attention_backward_delta
from kernels.backward.attention import attention_backward_dkdv
from kernels.backward.attention import attention_backward_dq
from kernels.streaming.attention import continuous_gqa_decode
from kernels.streaming.attention import flash_attention_fwd
from kernels.streaming.attention import flash_varlen_gqa_prefill
from kernels.streaming.attention import varlen_gqa_decode_with_sink_logits
from kernels.streaming.attention_specialized import block_causal_attention_fwd
from kernels.streaming.attention_specialized import native_sparse_attention_fwd
from kernels.streaming.attention_specialized import varlen_block_causal_attention_fwd
from kernels.streaming.block_sparse_attention import block_sparse_gqa_decode_combine
from kernels.streaming.block_sparse_attention import block_sparse_gqa_decode_partials
from kernels.streaming.mla import paged_mla_decode

from ...measurement import compile_single
from ...measurement import functional_launch
from ...model import Context
from ...model import PreparedComparison
from ...model import PreparedLaunch
from ...model import Tolerance
from .common import runtime_module
from .common import source_from_runtime
from .. import implementation_gap


def dense_flash_attention(context: Context) -> PreparedComparison:
    batch, sequence, heads, dimension = 4, 4096, 32, 128
    source_shape = (batch, sequence, heads, dimension)
    q_source = torch.randn(source_shape, device="cuda", dtype=torch.float16)
    k_source = torch.randn_like(q_source)
    v_source = torch.randn_like(q_source)
    q = q_source.permute(0, 2, 1, 3).contiguous()
    k = k_source.permute(0, 2, 1, 3).contiguous()
    v = v_source.permute(0, 2, 1, 3).contiguous()
    scale = 1.0 / math.sqrt(dimension)
    _, generated = compile_single(
        context,
        flash_attention_fwd,
        (q, k, v, scale),
        constexprs={"CAUSAL": True},
    )
    _, source_module = source_from_runtime(
        context,
        "source/tilelang/tilelang/attention/flash_forward_bshd/example_mha_fwd_bshd_runtime.py",
        "intent_v2_tilelang_dense_flash_attention",
    )
    source_kernel = source_module.flashattn(
        batch,
        heads,
        sequence,
        dimension,
        True,
        block_M=128,
        block_N=128,
        num_stages=1,
        threads=128,
    )
    source = functional_launch(
        lambda: source_kernel(q_source, k_source, v_source)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=5e-2, rtol=2e-2),
        cuda_graph=False,
    )


def varlen_gqa_prefill(context: Context) -> PreparedComparison:
    lengths_list = (4096, 3968, 3840, 3712, 3584, 3456, 3328, 3200)
    batch, heads, key_heads, dimension = 8, 32, 8, 128
    lengths = torch.tensor(lengths_list, device="cuda", dtype=torch.int32)
    offsets = torch.tensor(
        (0, *itertools.accumulate(lengths_list)), device="cuda", dtype=torch.int32
    )
    total = sum(lengths_list)
    q = torch.randn((total, heads, dimension), device="cuda", dtype=torch.float16)
    k = torch.randn(
        (total, key_heads, dimension), device="cuda", dtype=torch.float16
    )
    v = torch.randn_like(k)
    scale = 1.0 / math.sqrt(dimension)
    _, generated = compile_single(
        context,
        flash_varlen_gqa_prefill,
        (q, k, v, lengths, offsets, scale),
        constexprs={"HEAD_GROUP": heads // key_heads},
    )
    runtime = runtime_module(
        context,
        "source/tilelang/tilelang/attention/flash_forward_varlen/example_gqa_fwd_varlen_runtime.py",
        "intent_v2_tilelang_varlen_gqa_runtime",
    )
    source_kernel = runtime.source.flashattn(
        batch,
        heads // key_heads,
        total,
        total,
        heads,
        dimension,
        True,
        block_M=64,
        block_N=64,
        num_stages=2,
        threads=128,
    )
    source = functional_launch(
        lambda: source_kernel(q, k, v, offsets, offsets, max(lengths_list))
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=5e-2, rtol=2e-2),
        cuda_graph=False,
    )


def gqa_decode(context: Context) -> PreparedComparison:
    batch, heads, key_heads, sequence, dimension = 32, 32, 8, 8192, 128
    q = torch.randn((batch, heads, dimension), device="cuda", dtype=torch.float16)
    k = torch.randn(
        (batch, sequence, key_heads, dimension),
        device="cuda",
        dtype=torch.float16,
    )
    v = torch.randn_like(k)
    mask = torch.ones(
        (batch, sequence, key_heads), device="cuda", dtype=torch.uint8
    )
    scale = 1.0 / math.sqrt(dimension)
    _, generated = compile_single(
        context,
        continuous_gqa_decode,
        (q, k, v, mask, scale),
        constexprs={"HEAD_GROUP": heads // key_heads},
    )
    runtime = runtime_module(
        context,
        "source/tilelang/tilelang/attention/gqa_decode/example_gqa_decode_runtime.py",
        "intent_v2_tilelang_gqa_decode_runtime",
    )
    source = functional_launch(lambda: runtime.upstream((q, k, v, mask, scale)))
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=5e-2, rtol=2e-2),
        cuda_graph=False,
    )


def varlen_gqa_logits(context: Context) -> PreparedComparison:
    batch, heads, key_heads, dimension, maximum = 16, 32, 8, 64, 4096
    lengths_list = tuple(maximum - index * 64 for index in range(batch))
    offsets = torch.tensor(
        (0, *itertools.accumulate(lengths_list)), device="cuda", dtype=torch.int32
    )
    total = sum(lengths_list)
    q = torch.randn((batch, heads, dimension), device="cuda", dtype=torch.float16)
    k = torch.randn(
        (total, key_heads, dimension), device="cuda", dtype=torch.float16
    )
    v = torch.randn_like(k)
    sink = torch.zeros((heads,), device="cuda", dtype=torch.float32)
    scale = 1.0 / math.sqrt(dimension)
    _, generated = compile_single(
        context,
        varlen_gqa_decode_with_sink_logits,
        (q, k, v, offsets, sink, scale),
        constexprs={
            "HEAD_GROUP": heads // key_heads,
            "MAX_BLOCKS": maximum // 64,
        },
    )
    runtime = runtime_module(
        context,
        "source/tilelang/tilelang/attention/gqa_decode_varlen_logits/example_gqa_decode_varlen_logits_runtime.py",
        "intent_v2_tilelang_varlen_gqa_logits_runtime",
    )
    source_kernel = runtime.source.flashattn(
        batch,
        heads,
        key_heads,
        maximum,
        total,
        dimension,
        False,
        block_N=64,
        block_H=64,
        num_split=1,
        num_stages=2,
        threads=128,
    )
    source = functional_launch(lambda: source_kernel(q, k, v, offsets, sink))
    return PreparedComparison(
        generated,
        source,
        (Tolerance(atol=5e-2, rtol=2e-2), Tolerance(atol=5e-2, rtol=2e-2)),
        cuda_graph=False,
    )


def block_causal(context: Context) -> PreparedComparison:
    batch, sequence, heads, dimension = 2, 4096, 16, 128
    q = torch.randn(
        (batch, sequence, heads, dimension), device="cuda", dtype=torch.float16
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    scale = 1.0 / math.sqrt(dimension)
    _, generated = compile_single(
        context,
        block_causal_attention_fwd,
        (q, k, v, scale),
        constexprs={"BLOCK": 64},
    )
    support = runtime_module(
        context,
        "source/tilelang/tilelang/support/runtime.py",
        "intent_v2_tilelang_runtime_support",
    )
    source_module = support.load_source(
        context.project_root
        / "source/tilelang/tilelang/attention/block_causal/block_causal_attention.py",
        "intent_v2_tilelang_block_causal",
    )
    source = functional_launch(
        lambda: source_module.block_causal_attention(q, k, v, 64)
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=5e-2, rtol=2e-2),
        cuda_graph=False,
    )


def varlen_block_causal(context: Context) -> PreparedComparison:
    lengths = (4096, 3840, 3584, 3328)
    total, heads, dimension = sum(lengths), 16, 128
    offsets = torch.tensor(
        (0, *itertools.accumulate(lengths)), device="cuda", dtype=torch.int32
    )
    q = torch.randn((total, heads, dimension), device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    scale = 1.0 / math.sqrt(dimension)
    _, generated = compile_single(
        context,
        varlen_block_causal_attention_fwd,
        (q, k, v, offsets, scale),
        constexprs={"BLOCK": 64},
    )
    support = runtime_module(
        context,
        "source/tilelang/tilelang/support/runtime.py",
        "intent_v2_tilelang_runtime_support",
    )
    source_module = support.load_source(
        context.project_root
        / "source/tilelang/tilelang/attention/block_causal_varlen/block_causal_attention_varlen.py",
        "intent_v2_tilelang_varlen_block_causal",
    )
    source = functional_launch(
        lambda: source_module.block_causal_attention_varlen(
            q, k, v, offsets, 64, max_seqlen=max(lengths), block_size=64
        )
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=5e-2, rtol=2e-2),
        cuda_graph=False,
    )


def _native_sparse(context: Context, *, decode: bool) -> PreparedComparison:
    if decode:
        batch, query, sequence, heads, key_heads, dimension = 8, 1, 8192, 32, 2, 128
        selected, block = 32, 128
        source_path = "source/tilelang/tilelang/attention/native_sparse_decode/example_tilelang_nsa_decode.py"
        name = "intent_v2_tilelang_native_sparse_decode"
    else:
        batch, query, sequence, heads, key_heads, dimension = 2, 4096, 4096, 32, 4, 128
        selected, block = 64, 64
        source_path = "source/tilelang/tilelang/attention/native_sparse_forward/example_tilelang_nsa_fwd.py"
        name = "intent_v2_tilelang_native_sparse_forward"
    q = torch.randn(
        (batch, query, heads, dimension), device="cuda", dtype=torch.float16
    )
    k = torch.randn(
        (batch, sequence, key_heads, dimension),
        device="cuda",
        dtype=torch.float16,
    )
    v = torch.randn_like(k)
    block_ids = (
        torch.arange(selected, device="cuda", dtype=torch.int32)
        .reshape(1, 1, 1, selected)
        .expand(batch, query, key_heads, selected)
        .contiguous()
    )
    scale = 1.0 / math.sqrt(dimension)
    _, generated = compile_single(
        context,
        native_sparse_attention_fwd,
        (q, k, v, block_ids, scale),
        constexprs={"HEAD_GROUP": heads // key_heads, "BLOCK": block},
    )
    support = runtime_module(
        context,
        "source/tilelang/tilelang/support/runtime.py",
        "intent_v2_tilelang_runtime_support",
    )
    source_file = context.project_root / source_path
    source_module = support.load_source(
        source_file,
        name,
        aliases=(("reference", source_file.with_name("reference.py")),),
    )
    kwargs = {
        "dim": dimension,
        "block_size": block,
        "groups": heads // key_heads,
        "selected_blocks": selected,
    }
    if not decode:
        kwargs["is_causal"] = True
    source = functional_launch(
        lambda: source_module.native_sparse_attention(q, k, v, block_ids, **kwargs)
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=5e-2, rtol=2e-2),
        cuda_graph=False,
    )


def native_sparse_forward(context: Context) -> PreparedComparison:
    return _native_sparse(context, decode=False)


def native_sparse_decode(context: Context) -> PreparedComparison:
    return _native_sparse(context, decode=True)


def block_sparse_gqa_decode(context: Context) -> PreparedComparison:
    batch, query_heads, key_heads, sequence, dimension = 8, 32, 8, 8192, 128
    block_size, selected_blocks, splits = 32, 128, 4
    q = torch.randn(
        (batch, query_heads, dimension), device="cuda", dtype=torch.float16
    )
    k = torch.randn(
        (batch, sequence, key_heads, dimension),
        device="cuda",
        dtype=torch.float16,
    )
    v = torch.randn_like(k)
    selected = torch.arange(
        sequence // block_size - 1,
        sequence // block_size - selected_blocks - 1,
        -1,
        device="cuda",
        dtype=torch.int32,
    ).view(1, 1, selected_blocks).expand(batch, key_heads, selected_blocks).contiguous()
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
        (q, k, v, selected, lengths, split_offsets, scale),
        constexprs={
            "HEAD_GROUP": query_heads // key_heads,
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

    generated = PreparedLaunch(generated_launch, combined.outputs)
    runtime = runtime_module(
        context,
        "source/tilelang/tilelang/attention/blocksparse_gqa_decode_varlen/example_tilelang_sparse_gqa_decode_varlen_indice_runtime.py",
        "intent_v2_tilelang_block_sparse_gqa_runtime",
    )
    source_kernel = runtime.source.flashattn(
        batch,
        query_heads,
        key_heads,
        dimension,
        dimension,
        block_N=block_size,
        block_H=64,
        num_stages=2,
        threads=128,
    )
    source_lse = torch.empty(
        (batch, query_heads, splits), device="cuda", dtype=torch.float32
    )
    source_partial = torch.empty(
        (batch, query_heads, splits, dimension),
        device="cuda",
        dtype=torch.float32,
    )
    source_output = source_kernel(
        q, k, v, selected, lengths, source_lse, source_partial
    )
    source_executable = source_kernel.adapter._get_executable()

    def source_launch():
        source_executable(
            q,
            k,
            v,
            selected,
            lengths,
            source_lse,
            source_partial,
            source_output,
        )

    source = PreparedLaunch(source_launch, lambda: source_output)
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=5e-2, rtol=5e-2),
        cuda_graph=False,
    )


def paged_mla(context: Context) -> PreparedComparison:
    batch, query_heads, key_heads = 32, 128, 1
    sequence, value_dimension, rope_dimension = 8192, 128, 64
    page_size = 64
    pages_per_sequence = sequence // page_size
    pages = batch * pages_per_sequence
    q_latent = torch.randn(
        (batch, query_heads, value_dimension),
        device="cuda",
        dtype=torch.float16,
    )
    q_rope = torch.randn(
        (batch, query_heads, rope_dimension),
        device="cuda",
        dtype=torch.float16,
    )
    latent_cache = torch.randn(
        (pages, page_size, key_heads, value_dimension),
        device="cuda",
        dtype=torch.float16,
    )
    rope_cache = torch.randn(
        (pages, page_size, key_heads, rope_dimension),
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
    scale = (value_dimension + rope_dimension) ** -0.5
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
        constexprs={
            "PAGE_SIZE": page_size,
            "HEAD_GROUP": query_heads // key_heads,
        },
    )
    runtime = runtime_module(
        context,
        "source/tilelang/tilelang/attention/mla_decode_paged/example_mla_decode_paged_runtime.py",
        "intent_v2_tilelang_paged_mla_runtime",
    )
    source_module = runtime.load_source()
    source_kernel = source_module.mla_decode_tilelang(
        batch,
        query_heads,
        key_heads,
        sequence,
        value_dimension,
        rope_dimension,
        64,
        64,
        1,
        page_size,
        scale,
    )
    source_lse = torch.empty(
        (batch, query_heads, 1), device="cuda", dtype=torch.float16
    )
    source_partial = torch.empty(
        (batch, query_heads, 1, value_dimension),
        device="cuda",
        dtype=torch.float16,
    )
    flat_latent = latent_cache.view(-1, key_heads, value_dimension)
    flat_rope = rope_cache.view(-1, key_heads, rope_dimension)
    source_output = source_kernel(
        q_latent,
        q_rope,
        flat_latent,
        flat_rope,
        page_table,
        lengths,
        source_lse,
        source_partial,
    )
    source_executable = source_kernel.adapter._get_executable()

    def source_launch():
        source_executable(
            q_latent,
            q_rope,
            flat_latent,
            flat_rope,
            page_table,
            lengths,
            source_lse,
            source_partial,
            source_output,
        )

    source = PreparedLaunch(source_launch, lambda: source_output)
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=5e-2, rtol=5e-2),
        cuda_graph=False,
    )


def gqa_attention_backward(context: Context) -> PreparedComparison:
    batch, sequence, query_heads, key_heads, dimension = 1, 4096, 32, 8, 64
    source_q = torch.randn(
        (batch, sequence, query_heads, dimension),
        device="cuda",
        dtype=torch.float16,
    ) * 0.5
    source_k = torch.randn(
        (batch, sequence, key_heads, dimension),
        device="cuda",
        dtype=torch.float16,
    ) * 0.5
    source_v = torch.randn_like(source_k) * 0.5
    q = source_q.permute(0, 2, 1, 3).contiguous()
    k = source_k.permute(0, 2, 1, 3).contiguous()
    v = source_v.permute(0, 2, 1, 3).contiguous()
    runtime = runtime_module(
        context,
        "source/tilelang/tilelang/attention/gqa_backward/example_gqa_bwd_runtime.py",
        "intent_v2_tilelang_gqa_backward_runtime",
    )
    source_module = runtime.load_source()
    forward = source_module.flashattn_fwd(
        batch,
        query_heads,
        sequence,
        dimension,
        dimension,
        True,
        128,
        64,
        query_heads // key_heads,
    )
    source_output, source_lse = forward(source_q, source_k, source_v)
    output = source_output.permute(0, 2, 1, 3).contiguous()
    lse = source_lse / math.log2(math.e)
    source_grad_output = torch.randn_like(source_output) * 0.05
    grad_output = source_grad_output.permute(0, 2, 1, 3).contiguous()
    scale = dimension**-0.5
    _, delta = compile_single(
        context, attention_backward_delta, (output, grad_output)
    )
    common = (q, k, v, grad_output, lse, delta.outputs(), scale)
    constexprs = {
        "HEAD_GROUP": query_heads // key_heads,
        "CAUSAL": True,
    }
    _, dkdv = compile_single(
        context, attention_backward_dkdv, common, constexprs=constexprs
    )
    _, dq = compile_single(
        context, attention_backward_dq, common, constexprs=constexprs
    )

    def generated_launch():
        delta.launch()
        dkdv.launch()
        dq.launch()

    generated = PreparedLaunch(
        generated_launch,
        lambda: (dq.outputs(), *dkdv.outputs()),
    )
    preprocess = source_module.flashattn_bwd_preprocess(
        batch, query_heads, sequence, dimension
    )
    source_delta = preprocess(source_output, source_grad_output)
    backward = source_module.flashattn_bwd_split(
        batch,
        query_heads,
        sequence,
        dimension,
        dimension,
        True,
        128,
        32,
        groups=query_heads // key_heads,
    )
    postprocess = source_module.flashattn_bwd_postprocess(
        batch, query_heads, sequence, dimension
    )
    source_grad_q = torch.zeros_like(source_q, dtype=torch.float32)
    source_grad_k_parts = torch.empty(
        (query_heads // key_heads, batch, sequence, key_heads, dimension),
        device="cuda",
        dtype=torch.float16,
    )
    source_grad_v_parts = torch.empty_like(source_grad_k_parts)
    source_state: dict[str, tuple[torch.Tensor, ...]] = {}

    def source_launch():
        source_grad_q.zero_()
        backward(
            source_q,
            source_k,
            source_v,
            source_grad_output,
            source_lse,
            source_delta,
            source_grad_q,
            source_grad_k_parts,
            source_grad_v_parts,
        )
        source_state["outputs"] = (
            postprocess(source_grad_q).permute(0, 2, 1, 3),
            source_grad_k_parts.sum(0).permute(0, 2, 1, 3),
            source_grad_v_parts.sum(0).permute(0, 2, 1, 3),
        )

    source_launch()
    source = PreparedLaunch(source_launch, lambda: source_state["outputs"])
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
    "block_sparse_gqa_decode": block_sparse_gqa_decode,
    "dense_flash_attention": dense_flash_attention,
    "varlen_gqa_prefill": varlen_gqa_prefill,
    "gqa_decode": gqa_decode,
    "varlen_gqa_decode_logits": varlen_gqa_logits,
    "block_causal_attention": block_causal,
    "varlen_block_causal_attention": varlen_block_causal,
    "native_sparse_attention_forward": native_sparse_forward,
    "native_sparse_attention_decode": native_sparse_decode,
    "paged_mla_decode": paged_mla,
    "persistent_mla_decode": implementation_gap(
        "the source is one cooperative persistent kernel with grid-wide "
        "synchronization; the existing Intent MLA entry is a different "
        "two-kernel split-K program"
    ),
    "gqa_attention_backward": gqa_attention_backward,
    "sparse_mla_backward": implementation_gap(
        "the backward requires indexed sparse probability recomputation and "
        "many-to-one dKV scatter accumulation, which the current DSL corpus "
        "does not express as this source pipeline"
    ),
}
