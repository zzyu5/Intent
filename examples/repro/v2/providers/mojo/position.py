import torch
import intent

from kernels.position.rope import PARTIAL_ROTARY_DIMENSION
from kernels.position.rope import rotary_embedding_bf16
from kernels.position.rope import rotary_qk_bf16_inplace
from kernels.position.rope import rotary_qk_inplace
from kernels.position.rope import rotary_qk_partial_inplace

from ...loading import load_module
from ...measurement import report_stage
from ...model import PreparedComparison, PreparedLaunch, Tolerance
from .common import configure_cpu_budget, prepare_host_comparison


def rope_qk(context):
    batch, sequence, query_heads, kv_heads, dimension = 4, 4096, 32, 8, 128
    query_initial = torch.randn(
        (batch, query_heads, sequence, dimension), dtype=torch.float16
    )
    key_initial = torch.randn(
        (batch, kv_heads, sequence, dimension), dtype=torch.float16
    )
    positions = torch.arange(sequence, dtype=torch.float32)
    inverse = 1.0 / (
        10000 ** (torch.arange(0, dimension, 2, dtype=torch.float32) / dimension)
    )
    frequency = torch.outer(positions, inverse)
    embedding = torch.cat((frequency, frequency), dim=-1)
    cosine = embedding.cos().unsqueeze(0).to(torch.float16)
    sine = embedding.sin().unsqueeze(0).to(torch.float16)

    report_stage("generated_compilation")
    artifact = intent.compile(
        rotary_qk_inplace,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
    )
    generated_query = query_initial.clone()
    generated_key = key_initial.clone()

    def restore_generated():
        generated_query.copy_(query_initial)
        generated_key.copy_(key_initial)

    generated = PreparedLaunch(
        lambda: artifact.run(generated_query, generated_key, cosine, sine),
        lambda: (generated_query, generated_key),
        prepare=restore_generated,
    )

    runtime = load_module(
        context.project_root / "source/pytorch/cpu_runtime.py",
        "intent_cpu_reference_rope_qk",
    )
    source_query = query_initial.clone()
    source_key = key_initial.clone()

    def source_launch():
        runtime.rope_qk(source_query, source_key, cosine, sine)

    def restore_source():
        source_query.copy_(query_initial)
        source_key.copy_(key_initial)

    source = PreparedLaunch(
        source_launch,
        lambda: (source_query, source_key),
        prepare=restore_source,
    )
    report_stage("adapter_preparation")
    return PreparedComparison(
        generated,
        source,
        (Tolerance(atol=2e-3, rtol=2e-3), Tolerance(atol=2e-3, rtol=2e-3)),
        cuda_graph=False,
        device_type="cpu",
        cpu_host_timing=True,
        note="既有 rope_qk f16 InOut；PyTorch CPU reference，单 NUMA 8 核；每次调用前恢复 query/key，恢复不计时；双方计完整 host 调用。",
    )


def rope_qk_partial(context):
    configure_cpu_budget()
    batch, sequence, query_heads, kv_heads, dimension = 4, 2048, 32, 8, 128
    rotary_dimension = PARTIAL_ROTARY_DIMENSION
    half = rotary_dimension // 2
    query_initial = torch.randn(
        (batch, query_heads, sequence, dimension), dtype=torch.float16
    ) * 0.1
    key_initial = torch.randn(
        (batch, kv_heads, sequence, dimension), dtype=torch.float16
    ) * 0.1
    angles = torch.randn((1, sequence, half), dtype=torch.float32)
    cosine_half = angles.cos().to(torch.float16)
    sine_half = angles.sin().to(torch.float16)
    cosine = torch.cat((cosine_half, cosine_half), dim=2)
    sine = torch.cat((sine_half, sine_half), dim=2)

    report_stage("generated_compilation")
    artifact = intent.compile(
        rotary_qk_partial_inplace,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
    )
    generated_query = query_initial.clone()
    generated_key = key_initial.clone()

    def generated_launch():
        artifact.run(generated_query, generated_key, cosine, sine)

    def restore_generated():
        generated_query.copy_(query_initial)
        generated_key.copy_(key_initial)

    generated = PreparedLaunch(
        generated_launch,
        lambda: (generated_query, generated_key),
        prepare=restore_generated,
    )
    runtime = load_module(
        context.project_root / "source/pytorch/cpu_runtime.py",
        "intent_cpu_reference_rope_qk_partial",
    )
    source_query = query_initial.clone()
    source_key = key_initial.clone()

    def source_launch():
        runtime.rope_qk_partial(source_query, source_key, cosine, sine)

    def restore_source():
        source_query.copy_(query_initial)
        source_key.copy_(key_initial)

    source = PreparedLaunch(
        source_launch,
        lambda: (source_query, source_key),
        prepare=restore_source,
    )
    report_stage("adapter_preparation")
    return PreparedComparison(
        generated,
        source,
        (Tolerance(atol=2e-3, rtol=2e-3), Tolerance(atol=2e-3, rtol=2e-3)),
        cuda_graph=False,
        device_type="cpu",
        cpu_host_timing=True,
        note="既有 partial Q/K rotary f16 InOut；仅旋转前64维，PyTorch CPU reference 保持原f32乘法后f16写回；每次恢复 query/key，恢复不计时；双方计完整host调用。",
    )


def rotary_embedding(context):
    configure_cpu_budget()
    batch, sequence, heads, dimension = 4, 4096, 32, 128
    values = torch.randn(
        (batch, sequence, heads, dimension), dtype=torch.bfloat16
    )
    positions = torch.arange(sequence, dtype=torch.float32)
    inverse = 1.0 / (
        10000 ** (torch.arange(0, dimension, 2, dtype=torch.float32) / dimension)
    )
    angles = torch.outer(positions, inverse)
    cosine, sine = angles.cos(), angles.sin()
    return prepare_host_comparison(
        context,
        rotary_embedding_bf16,
        (values, cosine, sine),
        "rotary_embedding_bf16",
        Tolerance(atol=5e-3, rtol=5e-2),
    )


def rope_qk_bf16(context):
    configure_cpu_budget()
    batch, sequence, query_heads, kv_heads, dimension = 2, 4096, 32, 8, 128
    query_initial = torch.randn(
        (batch, query_heads, sequence, dimension), dtype=torch.bfloat16
    )
    key_initial = torch.randn(
        (batch, kv_heads, sequence, dimension), dtype=torch.bfloat16
    )
    cosine = torch.randn((1, sequence, dimension), dtype=torch.bfloat16)
    sine = torch.randn_like(cosine)

    report_stage("generated_compilation")
    artifact = intent.compile(
        rotary_qk_bf16_inplace,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
    )
    generated_query = query_initial.clone()
    generated_key = key_initial.clone()

    def generated_launch():
        artifact.run(generated_query, generated_key, cosine, sine)

    def restore_generated():
        generated_query.copy_(query_initial)
        generated_key.copy_(key_initial)

    generated = PreparedLaunch(
        generated_launch,
        lambda: (generated_query, generated_key),
        prepare=restore_generated,
    )
    runtime = load_module(
        context.project_root / "source/pytorch/cpu_runtime.py",
        "intent_cpu_reference_rope_qk_bf16",
    )
    source_query = query_initial.clone()
    source_key = key_initial.clone()

    def source_launch():
        runtime.rope_qk_bf16(source_query, source_key, cosine, sine)

    def restore_source():
        source_query.copy_(query_initial)
        source_key.copy_(key_initial)

    source = PreparedLaunch(
        source_launch,
        lambda: (source_query, source_key),
        prepare=restore_source,
    )
    report_stage("adapter_preparation")
    return PreparedComparison(
        generated,
        source,
        (Tolerance(atol=2e-2, rtol=1e-2), Tolerance(atol=2e-2, rtol=1e-2)),
        cuda_graph=False,
        device_type="cpu",
        cpu_host_timing=True,
        note="既有 Q/K rotary bf16 InOut；CPU reference 按 Intent 显式 f32 中间运算后转回 bf16，与原 CUDA source 的 bf16 中间舍入不同；每次恢复 query/key，恢复不计时；双方计完整host调用。",
    )


CASES = {"rope_qk": rope_qk, "rope_qk_partial": rope_qk_partial,
         "rotary_embedding_bf16": rotary_embedding,
         "rope_qk_bf16_inplace": rope_qk_bf16}
