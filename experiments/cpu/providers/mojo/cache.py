from __future__ import annotations

import intent
import torch

from kernels.cache.reshape_and_cache import BLOCKS
from kernels.cache.reshape_and_cache import BLOCK_SIZE
from kernels.cache.reshape_and_cache import HEADS
from kernels.cache.reshape_and_cache import HEAD_DIMENSION
from kernels.cache.reshape_and_cache import TOKENS
from kernels.cache.reshape_and_cache import reshape_and_cache
from kernels.variants.decomposition import reshape_key_cache, reshape_value_cache

from experiments._common.loading import load_module
from experiments._common.measurement import report_stage
from experiments._common.model import Context, PreparedComparison, PreparedLaunch, Tolerance
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
        tuning_config=context.tuning_config, options=context.compile_options,
    )
    runtime = load_module(
        context.project_root / "experiments/cpu/baselines/pytorch/cpu_runtime.py",
        "intent_cpu_reference_reshape_and_cache",
    )
    source_state: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def generated_prepare() -> None:
        generated_key_cache.zero_()
        generated_value_cache.zero_()

    def generated_launch() -> None:
        artifact.run(
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
            lambda: (generated_key_cache, generated_value_cache),
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


def reshape_and_cache_split(context):
    key = torch.randn((TOKENS, HEADS, HEAD_DIMENSION), dtype=torch.float16)
    value = torch.randn_like(key)
    slots = torch.randperm(BLOCKS * BLOCK_SIZE, dtype=torch.int64)[:TOKENS].to(torch.int32)
    report_stage("generated_compilation")
    key_program, value_program = (intent.compile(definition, target=context.target, compiler=context.compiler,
                                                 tuning_config=context.tuning_config, options=context.compile_options)
                                  for definition in (reshape_key_cache, reshape_value_cache))
    runtime = load_module(context.project_root / "experiments/cpu/baselines/pytorch/cpu_runtime.py", "intent_cpu_reference")

    def pipeline(key_cache, value_cache):
        key_program.run(key, slots, key_cache)
        value_program.run(value, slots, value_cache)

    def reference(key_cache, value_cache):
        runtime.reshape_and_cache(key, value, slots, key_cache, value_cache)

    def side(function):
        caches = tuple(torch.zeros((BLOCKS, BLOCK_SIZE, HEADS, HEAD_DIMENSION), dtype=torch.float16) for _ in range(2))

        def prepare():
            for cache in caches:
                cache.zero_()

        return PreparedLaunch(lambda: function(*caches), lambda: caches, prepare=prepare)

    report_stage("adapter_preparation")
    return PreparedComparison(side(pipeline), side(reference), Tolerance(atol=0.0),
        cuda_graph=False, device_type="cpu", cpu_host_timing=True,
        note="原T4096/H8/D128/2048x16 cache两kernel变体，随机唯一i32 slots，f16复制精确比较；双cache独立，清零不计时；计key/value两次kernel完整host调用。")


CASES = {"reshape_and_cache": reshape_and_cache_case, "reshape_and_cache_split": reshape_and_cache_split}
