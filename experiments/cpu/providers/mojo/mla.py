from __future__ import annotations

import intent
import torch

from kernels.streaming.mla import paged_mla_decode, paged_mla_decode_partials
from kernels.streaming.attention import mla_prefill
from kernels.streaming.mla import absorbed_mla_prefill, absorbed_mla_decode, splitk_mla_decode_partials
from kernels.streaming.mla import token_sparse_mla_prefill, token_sparse_mla_value_prefill, ABSORBED_MLA_SCALE
from kernels.streaming.splitk_reduce import splitk_attention_reduce_f16, splitk_attention_f32_to_f16_reduce

from experiments._common.loading import load_module
from experiments._common.measurement import report_stage
from experiments._common.model import Context, PreparedComparison, PreparedLaunch, Tolerance
from .common import configure_cpu_budget, prepare_host_comparison


def paged_mla(context: Context) -> PreparedComparison:
    configure_cpu_budget()
    batch, query_heads, key_heads = 32, 128, 1
    sequence, value_dimension, rope_dimension = 8192, 128, 64
    page_size = 64
    pages_per_sequence = sequence // page_size
    pages = batch * pages_per_sequence
    q_latent = torch.randn(
        (batch, query_heads, value_dimension), dtype=torch.float16
    )
    q_rope = torch.randn(
        (batch, query_heads, rope_dimension), dtype=torch.float16
    )
    latent_cache = torch.randn(
        (pages, page_size, key_heads, value_dimension), dtype=torch.float16
    )
    rope_cache = torch.randn(
        (pages, page_size, key_heads, rope_dimension), dtype=torch.float16
    )
    page_indices = torch.arange(pages, dtype=torch.int32)
    page_offsets = torch.arange(
        0,
        pages + 1,
        pages_per_sequence,
        dtype=torch.int32,
    )
    lengths = torch.full((batch,), sequence, dtype=torch.int32)
    scale = (value_dimension + rope_dimension) ** -0.5
    return prepare_host_comparison(
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
        "paged_mla_decode",
        Tolerance(atol=5e-2, rtol=5e-2),
        constexprs={
            "PAGE_SIZE": page_size,
            "HEAD_GROUP": query_heads // key_heads,
        },
    )


def dense_prefill(context):
    query = torch.randn((1, 128, 2048, 128), dtype=torch.float16)
    query_rope = torch.randn((1, 128, 2048, 64), dtype=torch.float16)
    key = torch.randn((1, 1, 2048, 128), dtype=torch.float16)
    key_rope = torch.randn((1, 1, 2048, 64), dtype=torch.float16)
    value = torch.randn_like(key)
    return prepare_host_comparison(context, mla_prefill, (query, query_rope, key, key_rope, value, 192**-0.5),
        "mla_prefill", Tolerance(atol=5e-2, rtol=2e-2), constexprs={"HEAD_GROUP": 128},
        note="原B1/HQ128/HK1/S2048/D128/R64 f16 causal MLA，生成端保留f16概率转换，CPU数学参考f32概率，最终f16，原容差。")


def absorbed_prefill(context):
    query = torch.randn((1, 512, 8, 512), dtype=torch.float16) * 0.1
    query_rope = torch.randn((1, 512, 8, 64), dtype=torch.float16) * 0.1
    cache = torch.randn((1, 512, 512), dtype=torch.float16) * 0.1
    cache_rope = torch.randn((1, 512, 64), dtype=torch.float16) * 0.1
    return prepare_host_comparison(context, absorbed_mla_prefill,
        (query, query_rope, cache, cache_rope, ABSORBED_MLA_SCALE), "absorbed_mla_prefill", Tolerance(atol=4e-2),
        note="原B1/S512/H8/C512/R64 f16 *.1，scale沿作者1/sqrt(NOPE128+R64)，不改为latent宽度；CPU复用原causal数学参考，生成端保留内部f16转换。")


def _absorbed_decode_inputs():
    query = torch.randn((8, 64, 512), dtype=torch.float16)
    query_rope = torch.randn((8, 64, 64), dtype=torch.float16)
    cache = torch.randn((8, 8192, 512), dtype=torch.float16)
    cache_rope = torch.randn((8, 8192, 64), dtype=torch.float16)
    return query, query_rope, cache, cache_rope, 576**-0.5


def absorbed_decode(context):
    return prepare_host_comparison(context, absorbed_mla_decode, _absorbed_decode_inputs(),
        "absorbed_mla_decode", Tolerance(atol=5e-2, rtol=2e-2),
        note="原B8/H64/S8192/C512/R64 f16 absorbed MLA decode；比较作者output ABI，CPU数学参考f32概率，生成端保留f16概率，原容差。")


def splitk_decode(context):
    arguments = _absorbed_decode_inputs()
    report_stage("generated_compilation")
    partials = intent.compile(splitk_mla_decode_partials, target=context.target, compiler=context.compiler,
                              tuning_config=context.tuning_config, constexprs={"SPLITS": 16, "SPLIT_SIZE": 512})
    reduction = intent.compile(splitk_attention_reduce_f16, target=context.target, compiler=context.compiler,
                               tuning_config=context.tuning_config)
    runtime = load_module(context.project_root / "experiments/cpu/baselines/pytorch/cpu_runtime.py", "intent_cpu_reference")
    generated, source = {}, {}

    def launch():
        lse, partial = partials.run(*arguments)
        generated["output"] = reduction.run(partial, lse)

    def reference():
        source["output"] = runtime.absorbed_mla_decode(*arguments)

    report_stage("adapter_preparation")
    return PreparedComparison(PreparedLaunch(launch, lambda: generated["output"]),
        PreparedLaunch(reference, lambda: source["output"]), Tolerance(atol=1e-1, rtol=5e-2),
        cuda_graph=False, device_type="cpu", cpu_host_timing=True,
        note="原B8/H64/S8192/C512/R64 f16 split512/P16 MLA完整partial+reduce，含workspace分配；partial f16/LSE f32，CPU完整attention数学参考f32，最终f16，原容差。")


def paged_mla_partials(context):
    query = torch.randn((8, 128, 512), dtype=torch.float16)
    query_rope = torch.randn((8, 128, 64), dtype=torch.float16)
    cache = torch.randn((4096, 16, 1, 512), dtype=torch.float16)
    cache_rope = torch.randn((4096, 16, 1, 64), dtype=torch.float16)
    offsets = torch.arange(0, 4097, 512, dtype=torch.int32)
    indices = torch.arange(4096, dtype=torch.int32)
    lengths = torch.full((8,), 8192, dtype=torch.int32)
    arguments = (query, query_rope, cache, cache_rope, offsets, indices, lengths, 576**-0.5)
    flat_arguments = (query, query_rope, cache.reshape(65536, 1, 512),
                      cache_rope.reshape(65536, 1, 64), offsets, indices, lengths, 576**-0.5)
    report_stage("generated_compilation")
    partials = intent.compile(paged_mla_decode_partials, target=context.target, compiler=context.compiler,
                              tuning_config=context.tuning_config,
                              constexprs={"PAGE_SIZE": 16, "HEAD_GROUP": 128, "SPLITS": 8})
    reduction = intent.compile(splitk_attention_f32_to_f16_reduce, target=context.target, compiler=context.compiler,
                               tuning_config=context.tuning_config)
    runtime = load_module(context.project_root / "experiments/cpu/baselines/pytorch/cpu_runtime.py", "intent_cpu_reference")
    generated, source = {}, {}

    def launch():
        lse, partial = partials.run(*flat_arguments)
        generated["output"] = reduction.run(partial, lse)

    def reference():
        source["output"] = runtime.paged_mla_decode(*arguments)

    report_stage("adapter_preparation")
    return PreparedComparison(PreparedLaunch(launch, lambda: generated["output"]),
        PreparedLaunch(reference, lambda: source["output"]), Tolerance(atol=5e-2, rtol=5e-2),
        cuda_graph=False, device_type="cpu", cpu_host_timing=True,
        note="原B8/HQ128/HK1/K8192/C512/R64/page16/split8 f16 MLA完整partial+reduce，含workspace分配；生成flattened page cache与f32 partial/LSE，CPU完整page-table attention数学参考，最终f16，原容差。")


def token_sparse_prefill(context):
    query = torch.randn((64, 8, 512), dtype=torch.bfloat16) * 0.1
    query_rope = torch.randn((64, 8, 64), dtype=torch.bfloat16) * 0.1
    cache = torch.randn((4096, 512), dtype=torch.bfloat16) * 0.1
    cache_rope = torch.randn((4096, 64), dtype=torch.bfloat16) * 0.1
    selected = torch.randint(0, 4096, (64, 256), dtype=torch.int32)
    selected[:, -2], selected[:, -1] = -1, 4096
    return prepare_host_comparison(context, token_sparse_mla_prefill,
        (query, query_rope, cache, cache_rope, selected, 576**-0.5), "token_sparse_mla_prefill",
        (Tolerance(atol=5e-2), Tolerance(atol=5e-3), Tolerance(atol=5e-3)),
        note="原Q64/H8/K4096/T256/C512/R64 BF16 *.1，selected末两项-1/K；沿原output/max/log2-LSE三项参考与容差，保留无效项mask，CPU参考f32概率。")


def token_sparse_value_prefill(context):
    query = torch.randn((1, 64, 2048, 128), dtype=torch.bfloat16)[0].permute(1, 0, 2).contiguous()
    query_rope = torch.randn((1, 64, 2048, 64), dtype=torch.bfloat16)[0].permute(1, 0, 2).contiguous()
    key = torch.randn((4096, 128), dtype=torch.bfloat16)
    value = torch.randn_like(key)
    key_rope = torch.randn((4096, 64), dtype=torch.bfloat16)
    selected = (torch.arange(2048, dtype=torch.int32)[:, None] - torch.arange(512, dtype=torch.int32)[None, :]).clamp_min(0)
    return prepare_host_comparison(context, token_sparse_mla_value_prefill,
        (query, query_rope, key, value, key_rope, selected, 192**-0.5), "token_sparse_mla_value_prefill",
        Tolerance(atol=1e-1, rtol=5e-2),
        note="原Q2048/H64/K4096/T512/D128/DV128/R64 BF16输入及clamp到0的历史selected tokens，重复项保留；CPU数学参考f32概率，生成端保留BF16概率，原容差。")


CASES = {
    "paged_mla_decode": paged_mla,
    "mla_prefill": dense_prefill,
    "absorbed_mla_prefill": absorbed_prefill,
    "absorbed_mla_decode": absorbed_decode,
    "splitk_mla_decode": splitk_decode,
    "paged_mla_decode_partials": paged_mla_partials,
    "token_sparse_mla_prefill": token_sparse_prefill,
    "token_sparse_mla_value_prefill": token_sparse_value_prefill,
}
