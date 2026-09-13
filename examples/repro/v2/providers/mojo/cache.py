from __future__ import annotations

import intent
import torch

from kernels.cache.reshape_and_cache import BLOCKS
from kernels.cache.reshape_and_cache import BLOCK_SIZE
from kernels.cache.reshape_and_cache import HEADS
from kernels.cache.reshape_and_cache import HEAD_DIMENSION
from kernels.cache.reshape_and_cache import TOKENS
from kernels.cache.reshape_and_cache import reshape_and_cache

from ...loading import load_module
from ...measurement import report_stage
from ...model import Context, PreparedComparison, PreparedLaunch, Tolerance
from .common import configure_cpu_budget


def reshape_and_cache_case(context: Context) -> PreparedComparison:
    configure_cpu_budget()
    key = torch.randn((TOKENS, HEADS, HEAD_DIMENSION), dtype=torch.float16)
    value = torch.randn_like(key)
    slots = (
        torch.randperm(BLOCKS * BLOCK_SIZE, dtype=torch.int64)[:TOKENS]
        .to(torch.int32)
    )
    cache_shape = (BLOCKS, BLOCK_SIZE, HEADS, HEAD_DIMENSION)
    generated_key_cache = torch.zeros(cache_shape, dtype=torch.float16)
    generated_value_cache = torch.zeros_like(generated_key_cache)
    source_key_cache = torch.zeros_like(generated_key_cache)
    source_value_cache = torch.zeros_like(generated_value_cache)

    report_stage("generated_compilation")
    artifact = intent.compile(
        reshape_and_cache,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
    )
    runtime = load_module(
        context.project_root / "source/pytorch/cpu_runtime.py",
        "intent_cpu_reference_reshape_and_cache",
    )
    generated_state: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    source_state: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def generated_prepare() -> None:
        generated_key_cache.zero_()
        generated_value_cache.zero_()

    def generated_launch() -> None:
        generated_state["output"] = artifact.run(
            key,
            value,
            slots,
            generated_key_cache,
            generated_value_cache,
        )

    def source_prepare() -> None:
        source_key_cache.zero_()
        source_value_cache.zero_()

    def source_launch() -> None:
        source_state["output"] = runtime.reshape_and_cache(
            key,
            value,
            slots,
            source_key_cache,
            source_value_cache,
        )

    report_stage("adapter_preparation")
    return PreparedComparison(
        PreparedLaunch(
            generated_launch,
            lambda: generated_state["output"],
            prepare=generated_prepare,
        ),
        PreparedLaunch(
            source_launch,
            lambda: source_state["output"],
            prepare=source_prepare,
        ),
        (Tolerance(atol=0.0), Tolerance(atol=0.0)),
        cuda_graph=False,
        device_type="cpu",
        cpu_host_timing=True,
        note="既有reshape-and-cache；T4096、H8、D128、2048x16 cache，随机i32 slot mapping；双f16 InOut cache独立恢复，完整host调用。",
    )


CASES = {"reshape_and_cache": reshape_and_cache_case}
