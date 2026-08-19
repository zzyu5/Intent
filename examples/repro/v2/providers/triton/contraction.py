from __future__ import annotations

import torch
import triton

from kernels.contraction.block_scaled import scaled_fp8_splitk_matmul
from kernels.contraction.gemm import Activation
from kernels.contraction.gemm import gemm
from kernels.ragged.grouped_gemm import ragged_grouped_gemm

from ...loading import load_module
from ...measurement import compile_single
from ...measurement import functional_launch
from ...model import Context
from ...model import PreparedComparison
from ...model import PreparedLaunch
from ...model import Tolerance


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
    _, generated = compile_single(context, ragged_grouped_gemm, (x, offsets, weight))
    runtime = _runtime(
        context,
        "source/triton/triton/gemm/grouped/08-grouped-gemm_runtime.py",
        "intent_v2_triton_grouped_gemm",
    )
    source_function = runtime.load_grouped_gemm()
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
    generated_stages = tuple(
        compile_single(
            context,
            gemm,
            (x, weight),
            constexprs={"ACTIVATION": Activation.NONE},
        )[1]
        for weight in weights
    )

    def generated_launch():
        for stage in generated_stages:
            stage.launch()

    generated = PreparedLaunch(
        launch=generated_launch,
        outputs=lambda: tuple(stage.outputs() for stage in generated_stages),
    )
    runtime = _runtime(
        context,
        "source/triton/xformers/gemm/tiled/tiled_matmul_kernels_runtime.py",
        "intent_v2_triton_qkv_projection",
    )
    source_module = runtime.load_source()
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
    generated_output = torch.zeros((m, n), device="cuda", dtype=torch.float16)
    _, generated_base = compile_single(
        context,
        scaled_fp8_splitk_matmul,
        (lhs, rhs, generated_output, 1.0, 1.0),
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
    block_m = block_n = 64
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
