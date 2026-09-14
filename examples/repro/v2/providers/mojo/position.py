import torch
import intent

from kernels.position.rope import PARTIAL_ROTARY_DIMENSION
from kernels.position.rope import rotary_embedding_bf16
from kernels.position.rope import rotary_embedding_flat
from kernels.position.rope import rotary_qk_bf16_inplace
from kernels.position.rope import rotary_qk_inplace
from kernels.position.rope import rotary_qk_partial_inplace
from kernels.position.rope_cache import padded_rope_cache_update as padded_rope_cache_update_definition

from ...loading import load_module
from ...measurement import report_stage
from ...model import PreparedComparison, PreparedLaunch, Tolerance
from .common import configure_cpu_budget, prepare_host_comparison


def rope_flat(context):
    values = torch.randn((65536, 128), dtype=torch.float16)
    cosine = torch.randn((2048, 64), dtype=torch.float16)
    sine = torch.randn_like(cosine)
    return prepare_host_comparison(context, rotary_embedding_flat, (values, cosine, sine),
                                   "rotary_embedding_flat", Tolerance(atol=1e-2), constexprs={"HEADS": 32})


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


def padded_rope_cache_update(context):
    configure_cpu_budget()
    batch, query_heads, kv_heads = 32, 32, 8
    cache_length, dimension = 8192, 128
    padded_length = cache_length + 1
    packed_input = torch.randn(
        (batch, query_heads + 2 * kv_heads, dimension), dtype=torch.float16
    )
    sequence_lengths = torch.full(
        (batch,), padded_length, dtype=torch.int32
    )
    query_rows = batch * query_heads
    cache_rows = batch * padded_length * kv_heads
    storage_shape = (query_rows + 2 * cache_rows, dimension)

    report_stage("generated_compilation")
    artifact = intent.compile(
        padded_rope_cache_update_definition,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
    )
    generated_storage = torch.empty(storage_shape, dtype=torch.float16)

    def generated_launch():
        artifact.run(
            packed_input,
            sequence_lengths,
            generated_storage,
            10000.0,
            1.0,
        )

    def generated_outputs():
        generated_query = generated_storage[:query_rows].view(
            batch, query_heads, dimension
        )
        generated_key_cache = generated_storage[
            query_rows : query_rows + cache_rows
        ].view(batch, padded_length, kv_heads, dimension)
        generated_value_cache = generated_storage[query_rows + cache_rows :].view(
            batch, padded_length, kv_heads, dimension
        )
        return (
            generated_query,
            generated_key_cache[:, cache_length],
            generated_value_cache[:, cache_length],
        )

    generated = PreparedLaunch(generated_launch, generated_outputs)

    runtime = load_module(
        context.project_root / "source/pytorch/cpu_runtime.py",
        "intent_cpu_reference_padded_rope_cache_update",
    )
    source_storage = torch.empty_like(generated_storage)

    def source_launch():
        runtime.padded_rope_cache_update(
            packed_input,
            sequence_lengths,
            source_storage,
            10000.0,
            1.0,
        )

    def source_outputs():
        source_query = source_storage[:query_rows].view(
            batch, query_heads, dimension
        )
        source_key_cache = source_storage[
            query_rows : query_rows + cache_rows
        ].view(batch, padded_length, kv_heads, dimension)
        source_value_cache = source_storage[query_rows + cache_rows :].view(
            batch, padded_length, kv_heads, dimension
        )
        return (
            source_query,
            source_key_cache[:, cache_length],
            source_value_cache[:, cache_length],
        )

    source = PreparedLaunch(source_launch, source_outputs)
    report_stage("adapter_preparation")
    return PreparedComparison(
        generated,
        source,
        (
            Tolerance(atol=2e-3, rtol=2e-3),
            Tolerance(atol=2e-3, rtol=2e-3),
            Tolerance(atol=0.0),
        ),
        cuda_graph=False,
        device_type="cpu",
        cpu_host_timing=True,
        note="既有 padded RoPE cache f16 InOut；保持原 B32/QH32/KVH8/D128/cache8192 输入、query/key/value 三个输出和原 tolerances；双方计完整 host 调用，输出 storage 分配不计时。",
    )


CASES = {"rotary_embedding_flat": rope_flat, "rope_qk": rope_qk, "rope_qk_partial": rope_qk_partial,
         "rotary_embedding_bf16": rotary_embedding,
         "rope_qk_bf16_inplace": rope_qk_bf16,
         "padded_rope_cache_update": padded_rope_cache_update}
