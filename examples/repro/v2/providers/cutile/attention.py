from __future__ import annotations

import math

import cuda.tile as ct
import torch

from kernels.backward.attention import attention_backward_delta
from kernels.backward.attention import attention_backward_dkdv
from kernels.backward.attention import attention_backward_dq
from kernels.streaming.attention import flash_gqa_attention_fwd
from kernels.streaming.attention import mla_prefill
from kernels.streaming.attention_specialized import attention_sink_prefill
from kernels.streaming.attention_specialized import gemma_gqa_prefill
from kernels.streaming.attention_specialized import sliding_window_gqa_prefill
from kernels.streaming.mla import absorbed_mla_decode
from kernels.streaming.mla import splitk_mla_decode_partials
from kernels.streaming.mla import token_sparse_mla_value_prefill
from kernels.streaming.splitk_reduce import splitk_attention_reduce
from kernels.streaming.splitk_reduce import splitk_attention_reduce_f16

from ...measurement import compile_single
from ...measurement import functional_launch
from ...model import Context
from ...model import PreparedComparison
from ...model import PreparedLaunch
from ...model import Tolerance
from .common import official_source
from .common import runtime_module
from .common import tilegym_source


def official_fmha(context: Context) -> PreparedComparison:
    batch, query_heads, key_heads, sequence, dimension = 4, 32, 8, 4096, 128
    q = torch.randn(
        (batch, query_heads, sequence, dimension),
        device="cuda",
        dtype=torch.float16,
    )
    k = torch.randn(
        (batch, key_heads, sequence, dimension),
        device="cuda",
        dtype=torch.float16,
    )
    v = torch.randn_like(k)
    scale = 1.0 / math.sqrt(dimension)
    _, generated = compile_single(
        context,
        flash_gqa_attention_fwd,
        (q, k, v, scale),
        constexprs={"HEAD_GROUP": query_heads // key_heads, "CAUSAL": True},
    )
    source_module = official_source(
        context,
        "source/cutile/cutile-python/attention/fmha/AttentionFMHA_runtime.py",
        "intent_v2_cutile_official_fmha",
    )
    source = functional_launch(
        lambda: source_module["cutile_fmha"](
            q,
            k,
            v,
            qk_scale=scale,
            query_group_size=query_heads // key_heads,
            causal=True,
        )
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=5e-2, rtol=2e-2),
        cuda_graph=True,
    )


def splitk_reduce(context: Context) -> PreparedComparison:
    batch, heads, splits, dimension = 8, 32, 16, 128
    partial = torch.randn(
        (batch, heads, splits, dimension),
        device="cuda",
        dtype=torch.bfloat16,
    )
    partial_lse = torch.randn(
        (batch, heads, splits), device="cuda", dtype=torch.float32
    )
    _, generated = compile_single(
        context, splitk_attention_reduce, (partial, partial_lse)
    )
    source_module = tilegym_source(
        context,
        "source/cutile/tilegym/attention/flash_decode/splitk_reduce.py",
        "splitk_attention_reduce",
    )
    source_output = torch.empty(
        (batch, heads, dimension), device="cuda", dtype=torch.bfloat16
    )

    def source_launch():
        source_module.splitk_reduce(
            partial,
            partial_lse,
            source_output,
            8192,
        )

    source_launch()
    source = PreparedLaunch(source_launch, lambda: source_output)
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=5e-2, rtol=2e-2),
        cuda_graph=True,
    )


def mla_prefill_case(context: Context) -> PreparedComparison:
    batch, heads, key_heads, sequence, dimension, rope = 1, 128, 1, 2048, 128, 64
    q = torch.randn(
        (batch, heads, sequence, dimension), device="cuda", dtype=torch.float16
    )
    qpe = torch.randn(
        (batch, heads, sequence, rope), device="cuda", dtype=torch.float16
    )
    k = torch.randn(
        (batch, key_heads, sequence, dimension),
        device="cuda",
        dtype=torch.float16,
    )
    v = torch.randn_like(k)
    kpe = torch.randn(
        (batch, 1, sequence, rope), device="cuda", dtype=torch.float16
    )
    scale = 1.0 / math.sqrt(dimension + rope)
    _, generated = compile_single(
        context,
        mla_prefill,
        (q, qpe, k, kpe, v, scale),
        constexprs={"HEAD_GROUP": heads // key_heads},
    )
    source_module = tilegym_source(
        context,
        "source/cutile/tilegym/attention/mla/mla.py",
        "mla_prefill",
    )
    source = functional_launch(
        lambda: source_module.tile_mla(
            q,
            k,
            v,
            qpe,
            kpe,
            is_causal=True,
            scaling=scale,
        )
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=5e-2, rtol=2e-2),
        cuda_graph=True,
    )


def attention_sink(context: Context) -> PreparedComparison:
    batch, sequence, key_heads, repeats, dimension = 1, 4096, 8, 4, 128
    query_heads = key_heads * repeats
    q = torch.randn(
        (batch, query_heads, sequence, dimension),
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        (batch, key_heads, sequence, dimension),
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)
    sinks = torch.randn((query_heads,), device="cuda", dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(dimension)
    _, generated = compile_single(
        context,
        attention_sink_prefill,
        (q, k, v, sinks, scale),
        constexprs={"HEAD_GROUP": repeats},
    )
    source_module = tilegym_source(
        context,
        "source/cutile/tilegym/attention/sink_prefill/attention_sink.py",
        "attention_sink_prefill",
    )
    q_source = (
        q.permute(0, 2, 1, 3)
        .contiguous()
        .view(batch, sequence, key_heads, repeats, dimension)
    )
    k_source = k.permute(0, 2, 1, 3).contiguous()
    v_source = v.permute(0, 2, 1, 3).contiguous()
    start = torch.zeros((1,), device="cuda", dtype=torch.int32)

    def source_call():
        output = source_module.attention_sink(
            q_source,
            k_source,
            v_source,
            sinks,
            scale,
            None,
            start,
        )
        return output.view(batch, sequence, query_heads, dimension).permute(
            0, 2, 1, 3
        )

    source = functional_launch(source_call)
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=5e-2, rtol=2e-2),
        cuda_graph=False,
    )


def gemma_prefill(context: Context) -> PreparedComparison:
    batch, sequence, query_heads, key_heads, dimension = 2, 4096, 32, 8, 128
    q = torch.randn(
        (batch, query_heads, sequence, dimension),
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        (batch, key_heads, sequence, dimension),
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)
    scale = 1.0 / math.sqrt(dimension)
    _, generated = compile_single(
        context,
        gemma_gqa_prefill,
        (q, k, v, scale),
        constexprs={
            "HEAD_GROUP": query_heads // key_heads,
            "WINDOW": 1024,
            "SOFT_CAP": 50.0,
        },
    )
    source_module = tilegym_source(
        context,
        "source/cutile/tilegym/attention/gemma_prefill/gemma_attention.py",
        "gemma_prefill",
    )
    source = functional_launch(
        lambda: source_module.gemma_attention_cutile(
            q,
            k,
            v,
            window_size=1024,
            soft_cap=50.0,
            is_causal=True,
            use_autotune=False,
        )
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=5e-2, rtol=2e-2),
        cuda_graph=True,
    )


def absorbed_mla(context: Context) -> PreparedComparison:
    batch, heads, sequence, latent, rope = 8, 64, 8192, 512, 64
    q = torch.randn((batch, heads, latent), device="cuda", dtype=torch.float16)
    qpe = torch.randn((batch, heads, rope), device="cuda", dtype=torch.float16)
    cache = torch.randn(
        (batch, sequence, latent), device="cuda", dtype=torch.float16
    )
    cache_pe = torch.randn(
        (batch, sequence, rope), device="cuda", dtype=torch.float16
    )
    scale = 1.0 / math.sqrt(latent + rope)
    _, generated = compile_single(
        context, absorbed_mla_decode, (q, qpe, cache, cache_pe, scale)
    )
    source_module = tilegym_source(
        context,
        "source/cutile/tilegym/attention/mla_decode/mla_decoding.py",
        "absorbed_mla_decode",
    )
    source = functional_launch(
        lambda: source_module.mla_decoding(q, qpe, cache, cache_pe, scale)[0]
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=5e-2, rtol=2e-2),
        cuda_graph=True,
    )


def sliding_window(context: Context) -> PreparedComparison:
    batch, sequence, query_heads, key_heads, dimension = 2, 4096, 32, 8, 128
    q = torch.randn(
        (batch, query_heads, sequence, dimension),
        device="cuda",
        dtype=torch.float16,
    )
    k = torch.randn(
        (batch, key_heads, sequence, dimension),
        device="cuda",
        dtype=torch.float16,
    )
    v = torch.randn_like(k)
    scale = 1.0 / math.sqrt(dimension)
    _, generated = compile_single(
        context,
        sliding_window_gqa_prefill,
        (q, k, v, scale),
        constexprs={"HEAD_GROUP": query_heads // key_heads, "WINDOW": 1024},
    )
    source_module = tilegym_source(
        context,
        "source/cutile/tilegym/attention/sliding_window/swa_attention.py",
        "sliding_window_attention",
    )
    source = functional_launch(
        lambda: source_module.tile_swa_attention(
            q, k, v, window_size=1024, is_causal=True
        )
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=5e-2, rtol=2e-2),
        cuda_graph=False,
    )


def attention_backward(context: Context) -> PreparedComparison:
    batch, query_heads, key_heads, sequence, dimension = 2, 8, 2, 1024, 64
    shape = (batch, query_heads, sequence, dimension)
    query = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.5
    key = torch.randn(
        (batch, key_heads, sequence, dimension),
        device="cuda",
        dtype=torch.float16,
    ) * 0.5
    value = torch.randn_like(key) * 0.5
    scale = dimension**-0.5
    repeated_key = key.repeat_interleave(query_heads // key_heads, dim=1)
    repeated_value = value.repeat_interleave(query_heads // key_heads, dim=1)
    scores = torch.matmul(
        query.float(), repeated_key.float().transpose(-1, -2)
    ) * scale
    causal = torch.ones(
        (sequence, sequence), device="cuda", dtype=torch.bool
    ).tril()
    scores.masked_fill_(~causal, -torch.inf)
    lse = torch.logsumexp(scores, dim=-1)
    source_lse = lse * math.log2(math.e)
    output = torch.matmul(torch.softmax(scores, dim=-1), repeated_value.float()).half()
    grad_output = torch.randn_like(output) * 0.05
    _, delta = compile_single(
        context, attention_backward_delta, (output, grad_output)
    )
    common = (query, key, value, grad_output, lse, delta.outputs(), scale)
    constexprs = {"HEAD_GROUP": query_heads // key_heads, "CAUSAL": True}
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
        launch=generated_launch,
        outputs=lambda: (dq.outputs(), *dkdv.outputs()),
    )
    runtime = runtime_module(
        context,
        "source/cutile/tilegym/attention/dense/attention_backward_runtime.py",
        "intent_v2_cutile_attention_backward_runtime",
    )
    source = functional_launch(
        lambda: runtime.upstream(
            (query, key, value, output, grad_output, source_lse, scale, True)
        )
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


def sparse_mla_prefill(context: Context) -> PreparedComparison:
    sequence, key_sequence, heads, topk, dimension, rope = 2048, 4096, 64, 512, 128, 64
    q_source = torch.randn(
        (1, heads, sequence, dimension), device="cuda", dtype=torch.bfloat16
    )
    k_source = torch.randn(
        (1, 1, key_sequence, dimension),
        device="cuda",
        dtype=torch.bfloat16,
    )
    v_source = torch.randn_like(k_source)
    qpe_source = torch.randn(
        (1, heads, sequence, rope), device="cuda", dtype=torch.bfloat16
    )
    kpe_source = torch.randn(
        (1, 1, key_sequence, rope), device="cuda", dtype=torch.bfloat16
    )
    positions = torch.arange(sequence, device="cuda", dtype=torch.int32)[:, None]
    selected = torch.maximum(
        positions - torch.arange(topk, device="cuda", dtype=torch.int32)[None, :],
        torch.zeros((), device="cuda", dtype=torch.int32),
    ).contiguous()
    source_indices = selected[None, :, None, :]
    q = q_source[0].permute(1, 0, 2).contiguous()
    qpe = qpe_source[0].permute(1, 0, 2).contiguous()
    k = k_source[0, 0]
    v = v_source[0, 0]
    kpe = kpe_source[0, 0]
    scale = 1.0 / math.sqrt(dimension + rope)
    _, generated = compile_single(
        context,
        token_sparse_mla_value_prefill,
        (q, qpe, k, v, kpe, selected, scale),
    )
    source_module = tilegym_source(
        context,
        "source/cutile/tilegym/attention/sparse_mla/sparse_mla.py",
        "sparse_mla_prefill",
    )

    def source_call():
        output = source_module.tile_sparse_mla(
            q_source,
            k_source,
            v_source,
            source_indices,
            qpe_source,
            kpe_source,
            is_causal=True,
            scaling=scale,
            kernel_configs={"TILE_H": 1, "TILE_N": 64},
        )
        return output[0].permute(1, 0, 2)

    source = functional_launch(source_call)
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=1e-1, rtol=5e-2),
        cuda_graph=False,
    )


def splitk_mla_decode(context: Context) -> PreparedComparison:
    batch, heads, sequence, latent, rope, split_size = 8, 64, 8192, 512, 64, 512
    splits = sequence // split_size
    query = torch.randn(
        (batch, heads, latent), device="cuda", dtype=torch.float16
    )
    query_rope = torch.randn(
        (batch, heads, rope), device="cuda", dtype=torch.float16
    )
    cache = torch.randn(
        (batch, sequence, latent), device="cuda", dtype=torch.float16
    )
    cache_rope = torch.randn(
        (batch, sequence, rope), device="cuda", dtype=torch.float16
    )
    split_offsets = torch.arange(
        0, sequence + 1, split_size, device="cuda", dtype=torch.int32
    )
    scale = 1.0 / math.sqrt(latent + rope)
    _, partials = compile_single(
        context,
        splitk_mla_decode_partials,
        (query, query_rope, cache, cache_rope, split_offsets, scale),
        constexprs={"SPLITS": splits},
    )
    partial_lse, partial_output = partials.outputs()
    _, reduction = compile_single(
        context,
        splitk_attention_reduce_f16,
        (partial_output, partial_lse),
    )

    def generated_launch():
        partials.launch()
        reduction.launch()

    generated = PreparedLaunch(generated_launch, reduction.outputs)
    source_module = tilegym_source(
        context,
        "source/cutile/tilegym/attention/mla_decode_split/mla_decoding_split_kv.py",
        "splitk_mla_decode",
        needs_utils=True,
        needs_splitk=True,
    )
    source_partial_output = torch.empty(
        (batch, heads, splits, latent), device="cuda", dtype=torch.float16
    )
    source_partial_lse = torch.empty(
        (batch, heads, splits), device="cuda", dtype=torch.float32
    )
    source_output = torch.empty_like(query)

    def source_launch():
        ct.launch(
            torch.cuda.current_stream(),
            ((heads + 15) // 16, batch, splits),
            source_module._naive_absorb_mla_transpose_kernel,
            (
                query,
                query_rope,
                cache,
                cache,
                cache_rope,
                source_partial_output,
                source_partial_lse,
                scale,
                batch,
                heads,
                sequence,
                splits,
                split_size,
                latent,
                16,
                128,
                rope,
                True,
            ),
        )
        source_module.splitk_reduce(
            source_partial_output,
            source_partial_lse,
            source_output,
            sequence,
        )

    source_launch()
    source = PreparedLaunch(
        source_launch,
        lambda: source_output,
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=1e-1, rtol=5e-2),
        cuda_graph=False,
    )


CASES = {
    "official_fmha": official_fmha,
    "splitk_attention_reduce": splitk_reduce,
    "mla_prefill": mla_prefill_case,
    "attention_sink_prefill": attention_sink,
    "gemma_prefill": gemma_prefill,
    "absorbed_mla_decode": absorbed_mla,
    "sliding_window_attention": sliding_window,
    "attention_backward": attention_backward,
    "sparse_mla_prefill": sparse_mla_prefill,
    "splitk_mla_decode": splitk_mla_decode,
}
