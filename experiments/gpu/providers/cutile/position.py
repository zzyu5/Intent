from __future__ import annotations

import torch

from kernels.position.rope import rotary_qk_bf16_inplace

from experiments._common.measurement import compile_single
from experiments._common.measurement import initial_launch
from experiments._common.model import Context
from experiments._common.model import PreparedComparison
from experiments._common.model import PreparedLaunch
from experiments._common.model import Tolerance
from .common import tilegym_source


def rope_qk(context: Context) -> PreparedComparison:
    batch, sequence, query_heads, key_heads, dimension = 2, 4096, 32, 8, 128
    query = torch.randn(
        (batch, query_heads, sequence, dimension),
        device="cuda",
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        (batch, key_heads, sequence, dimension),
        device="cuda",
        dtype=torch.bfloat16,
    )
    cosine = torch.randn(
        (1, sequence, dimension), device="cuda", dtype=torch.bfloat16
    )
    sine = torch.randn_like(cosine)
    initial_query = query.clone()
    initial_key = key.clone()
    source_query = initial_query.clone()
    source_key = initial_key.clone()
    artifact, _ = compile_single(
        context,
        rotary_qk_bf16_inplace,
        (query, key, cosine, sine),
    )
    def generated_prepare():
        query.copy_(initial_query)
        key.copy_(initial_key)

    operator = torch.compile(artifact.as_torch_op("intent_gpu_benchmark::rope_qk"), fullgraph=True)
    generated_launch = lambda: operator(query, key, cosine, sine)
    generated_prepare()
    initial_launch(generated_launch, side="generated")
    generated_prepare()
    generated = PreparedLaunch(
        generated_launch,
        lambda: (query, key),
        generated_prepare,
    )
    source_module = tilegym_source(
        context,
        "experiments/gpu/baselines/cutile/tilegym/position/rope/rope.py",
        "rope_qk",
    )
    def source_prepare():
        source_query.copy_(initial_query)
        source_key.copy_(initial_key)

    def source_launch():
        source_module.apply_rope_base(source_query, source_key, cosine, sine)

    source_prepare()
    initial_launch(source_launch, side="source")
    source_prepare()
    source = PreparedLaunch(
        source_launch,
        lambda: (source_query, source_key),
        source_prepare,
    )
    return PreparedComparison(
        generated,
        source,
        (Tolerance(atol=2e-2, rtol=1e-2), Tolerance(atol=2e-2, rtol=1e-2)),
        cuda_graph=False,
        # Source multiplies/adds bf16 tiles; Intent promotes both products to f32.
        note="同算法；source 保留 bf16 中间舍入；Intent 通过 mutable custom op 与 torch.compile(fullgraph=True) 调用，计完整执行，含框架产生的状态处理",
    )


CASES = {"rope_qk": rope_qk}
