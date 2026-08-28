from __future__ import annotations

import torch
import triton

from kernels.contraction.block_scaled import scaled_fp8_splitk_matmul
from kernels.contraction.gemm import Activation
from kernels.contraction.gemm import gemm
from kernels.contraction.qkv import fused_qkv_projection
from kernels.ragged.grouped_gemm import ragged_grouped_gemm

from ...loading import load_module
from ...measurement import compile_single
from ...measurement import functional_launch
from ...measurement import TRITON_PARAMETER_OWNERSHIP_M
from ...measurement import TRITON_PARAMETER_OWNERSHIP_N
from ...measurement import TRITON_PARAMETER_REDUCTION
from ...measurement import TRITON_PARAMETER_RESIDENT_WORKERS
from ...measurement import TRITON_PARAMETER_TRAVERSAL_GROUP
from ...measurement import TRITON_PARAMETER_TRAVERSAL_WORKERS
from ...measurement import triton_parameter_value
from ...model import Context
from ...model import PreparedComparison
from ...model import PreparedLaunch
from ...model import Tolerance


_DENSE_GEMM_CONFIGS = {
    (128, 256, 64, 8, 3),
    (64, 256, 32, 4, 4),
    (128, 128, 32, 4, 4),
    (128, 64, 32, 4, 4),
    (64, 128, 32, 4, 4),
    (128, 32, 32, 4, 4),
    (64, 32, 32, 2, 5),
    (32, 64, 32, 2, 5),
    (128, 256, 128, 8, 3),
    (256, 128, 128, 8, 3),
    (256, 64, 128, 4, 4),
    (64, 256, 128, 4, 4),
    (128, 128, 128, 4, 4),
    (128, 64, 64, 4, 4),
    (64, 128, 64, 4, 4),
    (128, 32, 64, 4, 4),
}

_QKV_PROJECTION_CONFIGS = (
    {
        (32, 64, block_k, 2, num_stages)
        for block_k in (32, 64)
        for num_stages in range(2, 7)
    }
    | {
        (32, 128, block_k, 4, num_stages)
        for block_k in (32, 64)
        for num_stages in range(2, 7)
    }
    | {(64, 128, block_k, 4, 4) for block_k in (32, 64)}
)

_GROUPED_GEMM_CONFIGS = {
    (128, 128, 64),
    (64, 128, 64),
}


def _runtime(context: Context, path: str, name: str):
    return load_module(context.project_root / path, name)


def dense_gemm(context: Context) -> PreparedComparison:
    m, k, n = 4096, 4096, 14336
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    _, generated = compile_single(
        context,
        gemm,
        (a, b),
        constexprs={"ACTIVATION": Activation.NONE},
        triton_config_filter=lambda config: (
            triton_parameter_value(config, TRITON_PARAMETER_OWNERSHIP_M),
            triton_parameter_value(config, TRITON_PARAMETER_OWNERSHIP_N),
            triton_parameter_value(config, TRITON_PARAMETER_REDUCTION),
            config.num_warps,
            config.num_stages,
        )
        in _DENSE_GEMM_CONFIGS
        and triton_parameter_value(
            config, TRITON_PARAMETER_TRAVERSAL_GROUP
        )
        == 8
        and config.num_ctas == 1,
    )
    runtime = _runtime(
        context,
        "source/triton/triton/gemm/dense/03-matrix-multiplication_runtime.py",
        "intent_v2_triton_dense_gemm",
    )
    source_function = runtime.load_matmul()
    source = functional_launch(lambda: source_function(a, b))
    return PreparedComparison(generated, source, Tolerance(atol=1e-2, rtol=1e-2), cuda_graph=True)


def grouped_gemm(context: Context) -> PreparedComparison:
    experts, rows, hidden, intermediate = 8, 1024, 4096, 14336
    x = torch.randn(
        (experts * rows, hidden), device="cuda", dtype=torch.float16
    )
    weight = torch.randn(
        (experts, hidden, intermediate), device="cuda", dtype=torch.float16
    )
    offsets = torch.arange(
        0,
        (experts + 1) * rows,
        rows,
        device="cuda",
        dtype=torch.int32,
    )
    resident_workers = torch.cuda.get_device_properties("cuda").multi_processor_count
    _, generated = compile_single(
        context,
        ragged_grouped_gemm,
        (x, offsets, weight),
        triton_config_filter=lambda config: (
            triton_parameter_value(config, TRITON_PARAMETER_OWNERSHIP_M),
            triton_parameter_value(config, TRITON_PARAMETER_OWNERSHIP_N),
            triton_parameter_value(config, TRITON_PARAMETER_REDUCTION),
        )
        in _GROUPED_GEMM_CONFIGS
        and triton_parameter_value(
            config, TRITON_PARAMETER_TRAVERSAL_WORKERS
        )
        == 1
        and triton_parameter_value(
            config, TRITON_PARAMETER_RESIDENT_WORKERS
        )
        == resident_workers
        and config.num_warps == 4
        and config.num_stages == 3
        and config.num_ctas == 1,
    )
    runtime = _runtime(
        context,
        "source/triton/triton/gemm/grouped/08-grouped-gemm_runtime.py",
        "intent_v2_triton_grouped_gemm",
    )
    source_function = runtime.load_grouped_gemm()
    source_autotuner = source_function.__globals__["grouped_matmul_kernel"]
    source_configs = [
        config
        for config in source_autotuner.configs
        if (
            config.kwargs["BLOCK_SIZE_M"],
            config.kwargs["BLOCK_SIZE_N"],
            config.kwargs["BLOCK_SIZE_K"],
        )
        in _GROUPED_GEMM_CONFIGS
        and config.kwargs["NUM_SM"] == resident_workers
        and config.num_warps == 4
        and config.num_stages == 3
        and config.num_ctas == 1
    ]
    if len(source_configs) != len(_GROUPED_GEMM_CONFIGS):
        raise RuntimeError(
            "grouped GEMM source does not expose the generated/source common "
            "persistent candidate set"
        )
    source_autotuner.configs = source_configs
    source_autotuner.early_config_prune = None
    source_autotuner.perf_model = None
    source_autotuner.configs_top_k = 1.0
    group_x = list(x.view(experts, rows, hidden).unbind(0))
    group_weight = list(weight.unbind(0))
    state: dict[str, object] = {}

    def launch():
        state["outputs"] = source_function(group_x, group_weight)

    launch()
    source = PreparedLaunch(
        launch=launch,
        outputs=lambda: torch.cat(state["outputs"], dim=0),
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=2e-2, rtol=1e-2),
        cuda_graph=False,
    )


def qkv_projection(context: Context) -> PreparedComparison:
    tokens = hidden = projection = 4096
    x = torch.randn((tokens, hidden), device="cuda", dtype=torch.float16)
    weights = tuple(
        torch.randn((hidden, projection), device="cuda", dtype=torch.float16)
        for _ in range(3)
    )
    packed_weights = torch.stack(weights)
    _, generated_base = compile_single(
        context,
        fused_qkv_projection,
        (x, packed_weights),
        triton_config_filter=lambda config: (
            triton_parameter_value(config, TRITON_PARAMETER_OWNERSHIP_M),
            triton_parameter_value(config, TRITON_PARAMETER_OWNERSHIP_N),
            triton_parameter_value(config, TRITON_PARAMETER_REDUCTION),
            config.num_warps,
            config.num_stages,
        )
        in _QKV_PROJECTION_CONFIGS
        and config.num_ctas == 1,
    )
    generated = PreparedLaunch(
        launch=generated_base.launch,
        outputs=lambda: tuple(generated_base.outputs()[index] for index in range(3)),
    )
    runtime = _runtime(
        context,
        "source/triton/xformers/gemm/tiled/tiled_matmul_kernels_runtime.py",
        "intent_v2_triton_qkv_projection",
    )
    source_module = runtime.load_source()
    source_autotuner = source_module._xformers_tiled_matmul_kernel
    source_configs = [
        config
        for config in source_autotuner.configs
        if (
            config.kwargs["BLOCK_M"],
            config.kwargs["BLOCK_N"],
            config.kwargs["BLOCK_K"],
            config.num_warps,
            config.num_stages,
        )
        in _QKV_PROJECTION_CONFIGS
        and config.kwargs["SPLIT_K"] == 1
        and config.kwargs["GROUP_M"] == 8
        and config.num_ctas == 1
    ]
    if len(source_configs) != len(_QKV_PROJECTION_CONFIGS):
        raise RuntimeError(
            "QKV source does not expose the generated/source common candidate set"
        )
    source_autotuner.configs = source_configs
    source_autotuner.early_config_prune = None
    source_autotuner.perf_model = None
    source_autotuner.configs_top_k = 1.0
    source_outputs = tuple(
        torch.empty((tokens, projection), device="cuda", dtype=torch.float16)
        for _ in range(3)
    )
    a = [[x]]
    b = [[*weights]]
    c = [[*source_outputs]]

    def launch():
        source_module._launch_triton_matmul(
            a,
            b,
            c,
            [tokens],
            [projection] * 3,
            [hidden],
        )

    launch()
    source = PreparedLaunch(launch=launch, outputs=lambda: source_outputs)
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=2e-2, rtol=1e-2),
        cuda_graph=False,
    )


def scaled_fp8_splitk(context: Context) -> PreparedComparison:
    m, n, k, splits, block = 4096, 14336, 4096, 4, 256
    iterations = k // (splits * block)
    lhs_storage = torch.randn(
        (m, k), device="cuda", dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    rhs_storage = torch.randn(
        (k, n), device="cuda", dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    lhs = lhs_storage.view(m, iterations, splits, block)
    rhs = (
        rhs_storage.view(iterations, splits, block, n)
    )
    block_m = block_n = 64
    generated_output = torch.zeros((m, n), device="cuda", dtype=torch.float16)
    _, generated_base = compile_single(
        context,
        scaled_fp8_splitk_matmul,
        (lhs, rhs, generated_output, 1.0, 1.0),
        triton_config_filter=lambda config: (
            triton_parameter_value(
                config, TRITON_PARAMETER_OWNERSHIP_N, dimension=1
            )
            == block_m
            and triton_parameter_value(
                config, TRITON_PARAMETER_OWNERSHIP_N, dimension=5
            )
            == block_n
            and config.num_warps == 8
            and config.num_stages == 3
            and config.num_ctas == 1
        ),
    )
    generated = PreparedLaunch(
        launch=generated_base.launch,
        outputs=lambda: generated_output,
        prepare=lambda: generated_output.zero_(),
    )

    source_module = load_module(
        context.project_root
        / "source/triton/meta-applied-ai/gemm/fp8_scaled/scaled_fp8_gemm.py",
        "intent_v2_triton_scaled_fp8_splitk",
    )
    source_output = torch.zeros_like(generated_output)
    grid = (
        triton.cdiv(m, block_m) * triton.cdiv(n, block_n),
        splits,
    )

    def launch():
        source_module.scaled_gemm_splitk[grid](
            lhs_storage,
            rhs_storage,
            source_output,
            lhs_storage.stride(0),
            lhs_storage.stride(1),
            rhs_storage.stride(0),
            rhs_storage.stride(1),
            source_output.stride(0),
            source_output.stride(1),
            1.0,
            1.0,
            m,
            n,
            k,
            block_m,
            block_n,
            block,
            splits,
            8,
            num_stages=3,
            num_warps=8,
        )

    launch()
    source = PreparedLaunch(
        launch=launch,
        outputs=lambda: source_output,
        prepare=lambda: source_output.zero_(),
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=5e-1, rtol=1e-2),
        cuda_graph=False,
    )


CASES = {
    "dense_gemm": dense_gemm,
    "grouped_gemm": grouped_gemm,
    "qkv_projection_pipeline": qkv_projection,
    "scaled_fp8_splitk_gemm": scaled_fp8_splitk,
}
