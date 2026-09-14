import math

import intent
import torch

from kernels.streaming.attention_specialized import gemma_gqa_decode_partials
from kernels.streaming.attention_f32 import causal_attention_f32, causal_linear_attention_f32
from kernels.streaming.attention import flash_attention_bf16_fwd
from kernels.streaming.attention import flash_attention_fwd, flash_gqa_attention_fwd
from kernels.streaming.attention import flash_attention_bias_fwd, continuous_gqa_decode
from kernels.streaming.attention import flash_varlen_attention_fwd, flash_varlen_gqa_prefill
from kernels.streaming.attention import VARLEN_BATCH, VARLEN_TOTAL_TOKENS
from kernels.streaming.block_sparse_attention import block_sparse_gqa_decode_combine
from kernels.streaming.block_sparse_attention import block_sparse_gqa_decode_partials
from kernels.streaming.paged_attention import paged_gqa_decode_partials
from kernels.streaming.splitk_reduce import splitk_attention_f32_to_f16_reduce
from kernels.streaming.splitk_reduce import splitk_attention_reduce
from ...loading import load_module
from ...measurement import report_stage
from ...model import Context, PreparedComparison, PreparedLaunch, Tolerance
from .common import configure_cpu_budget, prepare_comparison, prepare_host_comparison


F16_PRECISION_NOTE = "生成端保留作者的chunk概率f16转换后与V作f32累加；CPU数学reference使用f32概率，最终双方输出f16；沿用原production容差。"


def causal_attention(context):
    q = torch.randn((8, 128, 32), dtype=torch.float32)
    k, v = torch.randn_like(q), torch.randn_like(q)
    return prepare_comparison(context, causal_attention_f32, (q, k, v, 32 ** -0.5),
        "source/mojo/intentdsl/attention/causal/causal_runtime.py", Tolerance(2e-4, 1e-5),
        "独立 Mojo online attention source；沿用 CPU f32 contraction 容差；分段与归约括号可不同。")


def causal_linear_attention(context):
    q = torch.randn((8, 128, 32), dtype=torch.float32)
    k, v = torch.randn_like(q), torch.randn_like(q)
    return prepare_comparison(context, causal_linear_attention_f32, (q, k, v),
        "source/mojo/intentdsl/attention/linear/linear_runtime.py", Tolerance(2e-4, 1e-5),
        "独立 Mojo chunked linear attention source；输出包含最终状态；沿用 CPU f32 contraction 容差。")


def flash_attention_bf16(context):
    dimension = 128
    q = torch.randn((2, 32, 4096, dimension), dtype=torch.bfloat16)
    k = torch.randn((2, 8, 4096, dimension), dtype=torch.bfloat16)
    v = torch.randn_like(k)
    return prepare_host_comparison(
        context,
        flash_attention_bf16_fwd,
        (q, k, v, dimension**-0.5),
        "flash_attention_bf16_fwd",
        Tolerance(atol=5e-2, rtol=2e-2),
        constexprs={"HEAD_GROUP": 4, "CAUSAL": True},
    )


def flash_attention_f16(context):
    q = torch.randn((4, 32, 4096, 128), dtype=torch.float16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    return prepare_host_comparison(context, flash_attention_fwd, (q, k, v, 128**-0.5),
                                   "flash_attention_bf16_fwd", Tolerance(atol=2e-2, rtol=2e-2),
                                   constexprs={"CAUSAL": True}, note=F16_PRECISION_NOTE)


def flash_gqa_attention_f16(context):
    q = torch.randn((4, 32, 4096, 128), dtype=torch.float16)
    k = torch.randn((4, 8, 4096, 128), dtype=torch.float16)
    v = torch.randn_like(k)
    return prepare_host_comparison(context, flash_gqa_attention_fwd, (q, k, v, 128**-0.5),
                                   "flash_attention_bf16_fwd", Tolerance(atol=5e-2, rtol=2e-2),
                                   constexprs={"HEAD_GROUP": 4, "CAUSAL": True}, note=F16_PRECISION_NOTE)


def continuous_gqa(context):
    q = torch.randn((32, 32, 128), dtype=torch.float16)
    k = torch.randn((32, 8192, 8, 128), dtype=torch.float16)
    v = torch.randn_like(k)
    mask = torch.ones((32, 8192, 8), dtype=torch.uint8)
    return prepare_host_comparison(context, continuous_gqa_decode, (q, k, v, mask, 128**-0.5),
                                   "continuous_gqa_decode", Tolerance(atol=5e-2, rtol=2e-2),
                                   constexprs={"HEAD_GROUP": 4}, note=F16_PRECISION_NOTE)


def varlen_gqa_prefill(context):
    lengths = torch.tensor((4096, 3968, 3840, 3712, 3584, 3456, 3328, 3200), dtype=torch.int32)
    offsets = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).int()))
    q = torch.randn((29184, 32, 128), dtype=torch.float16)
    k = torch.randn((29184, 8, 128), dtype=torch.float16)
    v = torch.randn_like(k)
    return prepare_host_comparison(context, flash_varlen_gqa_prefill,
                                   (q, k, v, lengths, offsets, 128**-0.5),
                                   "flash_varlen_gqa_prefill", Tolerance(atol=5e-2, rtol=2e-2),
                                   constexprs={"HEAD_GROUP": 4}, note=F16_PRECISION_NOTE)


def _attention_run_only(context, definition, arguments, constexprs, note):
    report_stage("generated_compilation")
    artifact = intent.compile(definition, target=context.target, compiler=context.compiler,
                              tuning_config=context.tuning_config, constexprs=constexprs)
    state = {}

    def launch():
        state["output"] = artifact.run(*arguments)

    report_stage("adapter_preparation")
    return PreparedComparison(PreparedLaunch(launch, lambda: state["output"]), None, None,
        cuda_graph=False, status="run_only", device_type="cpu", cpu_host_timing=True, note=note)


def biased_attention(context):
    q = torch.randn((4, 32, 4096, 128), dtype=torch.float16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    bias = torch.randn((4, 32, 4096), dtype=torch.float32)
    return _attention_run_only(context, flash_attention_bias_fwd, (q, k, v, bias, 128**-0.5), {},
        "既有B4/H32/S4096/D128 f16 attention+bias f32 metadata；原provider没有可运行reference和容差，仅验证当前kernel完整host调用可运行，输出f16，非causal，不作数值或相对性能结论。")


def varlen_attention(context):
    lengths = torch.full((VARLEN_BATCH,), VARLEN_TOTAL_TOKENS // VARLEN_BATCH, dtype=torch.int32)
    lengths[:VARLEN_TOTAL_TOKENS % VARLEN_BATCH] += 1
    offsets = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).int()))
    q = torch.randn((VARLEN_TOTAL_TOKENS, 128), dtype=torch.float16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    return _attention_run_only(context, flash_varlen_attention_fwd,
        (q, k, v, lengths, offsets, 128**-0.5), {"CAUSAL": True},
        "按作者B8/U29114常量构造平衡分段，f16/D128/causal；该entry无既有production runner、reference或容差，仅验证完整host调用可运行，不作数值或相对性能结论。")


def gemma_decode(context: Context) -> PreparedComparison:
    configure_cpu_budget()
    batch, query_heads, key_heads, sequence, dimension, split_size = (
        32, 32, 8, 8192, 128, 256
    )
    splits = (sequence + split_size - 1) // split_size
    q = torch.randn(
        (batch, query_heads, 1, dimension), dtype=torch.bfloat16
    )
    k = torch.randn(
        (batch, key_heads, sequence, dimension), dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    scale = 1.0 / math.sqrt(dimension)
    report_stage("generated_compilation")
    partials = intent.compile(
        gemma_gqa_decode_partials,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
        constexprs={
            "HEAD_GROUP": query_heads // key_heads,
            "WINDOW": 1024,
            "SOFT_CAP": 50.0,
            "P": splits,
        },
    )
    reduction = intent.compile(
        splitk_attention_reduce,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
    )
    partial_lse = torch.empty(
        (batch, query_heads, splits), dtype=torch.float32
    )
    partial_output = torch.empty(
        (batch, query_heads, splits, dimension), dtype=torch.bfloat16
    )
    output = torch.empty(
        (batch, query_heads, dimension), dtype=torch.bfloat16
    )

    def generated_launch():
        partials(q, k, v, partial_lse, partial_output, scale)
        reduction(partial_output, partial_lse, output)

    generated = PreparedLaunch(
        generated_launch,
        lambda: output.unsqueeze(2),
    )
    report_stage("source_compilation")
    runtime = load_module(
        context.project_root / "source/pytorch/cpu_runtime.py",
        "intent_cpu_reference_gemma_decode",
    )
    source_state = {}

    def source_launch():
        source_state["output"] = runtime.gemma_decode(
            q,
            k,
            v,
            scale,
            window_size=1024,
            soft_cap=50.0,
            kv_len_per_split=split_size,
        )

    report_stage("adapter_preparation")
    return PreparedComparison(
        generated,
        PreparedLaunch(source_launch, lambda: source_state["output"]),
        Tolerance(atol=1e-1, rtol=5e-2),
        cuda_graph=False,
        device_type="cpu",
        cpu_host_timing=True,
        note="完整 Gemma GQA decode partial+split-k pipeline；生成端输出及 partial 预分配，CPU reference 为 PyTorch eager，计入其临时分配；原输入、输出、容差不变。",
    )


def paged_gqa_decode(context: Context) -> PreparedComparison:
    configure_cpu_budget()
    batch, query_heads, kv_heads, dimension = 16, 32, 8, 128
    sequence, page_size, splits = 8192, 16, 8
    pages_per_sequence = sequence // page_size
    pages_per_split = pages_per_sequence // splits
    pages = batch * pages_per_sequence
    q = torch.randn((batch, query_heads, dimension), dtype=torch.float16)
    key_cache = torch.randn(
        (pages, page_size, kv_heads, dimension), dtype=torch.float16
    )
    value_cache = torch.randn_like(key_cache)
    page_indices = torch.arange(pages, dtype=torch.int32)
    page_offsets = torch.arange(
        0, pages + 1, pages_per_sequence, dtype=torch.int32
    )
    split_offsets = torch.arange(
        0, pages + 1, pages_per_split, dtype=torch.int32
    )
    lengths = torch.full((batch,), sequence, dtype=torch.int32)
    scale = dimension**-0.5
    report_stage("generated_compilation")
    partials = intent.compile(
        paged_gqa_decode_partials,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
        constexprs={
            "PAGE_SIZE": page_size,
            "HEAD_GROUP": query_heads // kv_heads,
            "SPLITS": splits,
        },
    )
    reduction = intent.compile(
        splitk_attention_f32_to_f16_reduce,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
    )
    partial_lse = torch.empty(
        (batch, query_heads, splits), dtype=torch.float32
    )
    partial_output = torch.empty(
        (batch, query_heads, splits, dimension), dtype=torch.float32
    )
    output = torch.empty((batch, query_heads, dimension), dtype=torch.float16)

    def generated_launch():
        partials(
            q,
            key_cache,
            value_cache,
            page_offsets,
            page_indices,
            lengths,
            split_offsets,
            partial_lse,
            partial_output,
            scale,
        )
        reduction(partial_output, partial_lse, output)

    generated = PreparedLaunch(
        generated_launch,
        lambda: output,
    )
    report_stage("source_compilation")
    runtime = load_module(
        context.project_root / "source/pytorch/cpu_runtime.py",
        "intent_cpu_reference_paged_gqa_decode",
    )
    source_state = {}

    def source_launch():
        source_state["output"] = runtime.paged_gqa_decode(
            q,
            key_cache,
            value_cache,
            page_offsets,
            page_indices,
            lengths,
            split_offsets,
            scale,
            page_size=page_size,
        )

    report_stage("adapter_preparation")
    return PreparedComparison(
        generated,
        PreparedLaunch(source_launch, lambda: source_state["output"]),
        Tolerance(atol=5e-2, rtol=5e-2),
        cuda_graph=False,
        device_type="cpu",
        cpu_host_timing=True,
        note="完整 paged GQA partial+split-k pipeline；生成端输出及 partial 预分配，CPU reference 计入 page table 解码、临时分配和 attention。",
    )


def block_sparse_gqa_decode(context: Context) -> PreparedComparison:
    configure_cpu_budget()
    batch, query_heads, key_heads, sequence, dimension = 8, 32, 8, 8192, 128
    block_size, selected_blocks, splits = 32, 128, 4
    q = torch.randn((batch, query_heads, dimension), dtype=torch.float16)
    k = torch.randn(
        (batch, sequence, key_heads, dimension), dtype=torch.float16
    )
    v = torch.randn_like(k)
    selected = torch.arange(
        sequence // block_size - 1,
        sequence // block_size - selected_blocks - 1,
        -1,
        dtype=torch.int32,
    ).view(1, 1, selected_blocks).expand(
        batch, key_heads, selected_blocks
    ).contiguous()
    lengths = torch.full((batch,), sequence, dtype=torch.int32)
    split_offsets = torch.arange(
        0, selected_blocks + 1, selected_blocks // splits, dtype=torch.int32
    )
    scale = dimension**-0.5
    report_stage("generated_compilation")
    partials = intent.compile(
        block_sparse_gqa_decode_partials,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
        constexprs={
            "HEAD_GROUP": query_heads // key_heads,
            "BLOCK_SIZE": block_size,
            "SPLITS": splits,
        },
    )
    combine = intent.compile(
        block_sparse_gqa_decode_combine,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
        constexprs={"SPLITS": splits},
    )
    partial_lse = torch.empty(
        (batch, query_heads, splits), dtype=torch.float32
    )
    partial_output = torch.empty(
        (batch, query_heads, splits, dimension), dtype=torch.float32
    )
    output = torch.empty((batch, query_heads, dimension), dtype=torch.float16)

    def generated_launch():
        partials(
            q,
            k,
            v,
            selected,
            lengths,
            split_offsets,
            partial_lse,
            partial_output,
            scale,
        )
        combine(partial_lse, partial_output, output)

    generated = PreparedLaunch(
        generated_launch,
        lambda: output,
    )
    report_stage("source_compilation")
    runtime = load_module(
        context.project_root / "source/pytorch/cpu_runtime.py",
        "intent_cpu_reference_block_sparse_gqa_decode",
    )
    source_state = {}

    def source_launch():
        source_state["output"] = runtime.block_sparse_gqa_decode(
            q,
            k,
            v,
            selected,
            lengths,
            split_offsets,
            scale,
            block_size=block_size,
        )

    report_stage("adapter_preparation")
    return PreparedComparison(
        generated,
        PreparedLaunch(source_launch, lambda: source_state["output"]),
        Tolerance(atol=5e-2, rtol=5e-2),
        cuda_graph=False,
        device_type="cpu",
        cpu_host_timing=True,
        note="完整 block-sparse GQA partial+combine pipeline；保持原倒序 selected blocks；生成端输出及 partial 预分配，CPU reference 计入 block gather、临时分配和 attention。",
    )


CASES = {
    "causal_attention_f32": causal_attention,
    "causal_linear_attention_f32": causal_linear_attention,
    "flash_attention_bf16": flash_attention_bf16,
    "flash_attention_f16": flash_attention_f16,
    "flash_gqa_attention_f16": flash_gqa_attention_f16,
    "continuous_gqa_decode": continuous_gqa,
    "flash_varlen_gqa_prefill": varlen_gqa_prefill,
    "flash_attention_bias": biased_attention,
    "flash_varlen_attention": varlen_attention,
    "gemma_decode": gemma_decode,
    "paged_gqa_decode": paged_gqa_decode,
    "block_sparse_gqa_decode": block_sparse_gqa_decode,
}
