import torch
import intent

from kernels.position.rope import rotary_qk_inplace

from ...loading import load_module
from ...measurement import report_stage
from ...model import PreparedComparison, PreparedLaunch, Tolerance


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


CASES = {"rope_qk": rope_qk}
