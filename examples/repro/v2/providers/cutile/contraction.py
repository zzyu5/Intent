from __future__ import annotations

import torch

from kernels.contraction.batched_gemm import batched_gemm_nn
from kernels.contraction.block_scaled import block_scaled_matmul
from kernels.contraction.gemm import Activation
from kernels.contraction.gemm import bf16_gemm
from kernels.contraction.gemm import gemm
from kernels.ragged.grouped_gemm import ragged_grouped_gemm_bf16

from ...measurement import compile_single
from ...measurement import candidate_parameter_value
from ...measurement import TRITON_PARAMETER_OWNERSHIP_M
from ...measurement import TRITON_PARAMETER_OWNERSHIP_N
from ...measurement import TRITON_PARAMETER_REDUCTION
from ...measurement import TRITON_PARAMETER_TRAVERSAL_GROUP
from ...measurement import functional_launch
from ...model import Context
from ...model import PreparedComparison
from ...model import PreparedLaunch
from ...model import Tolerance
from .common import official_source
from .common import tilegym_source


def block_scaled_gemm(context: Context) -> PreparedComparison:
    m, k, n, block = 4096, 4096, 14336, 32
    source_module = official_source(
        context,
        "source/cutile/cutile-python/gemm/block_scaled/BlockScaledMatMul_runtime.py",
        "intent_v2_cutile_block_scaled_gemm",
    )
    lhs, lhs_scale = source_module.block_quantize(
        torch.randn((m, k), device="cuda", dtype=torch.float32), block
    )
    rhs_rows, rhs_scale_rows = source_module.block_quantize(
        torch.randn((n, k), device="cuda", dtype=torch.float32), block
    )
    rhs = rhs_rows.T
    rhs_scale = rhs_scale_rows.T
    _, generated = compile_single(
        context,
        block_scaled_matmul,
        (
            lhs.view(m, k // block, block),
            lhs_scale.view(torch.uint8),
            rhs.view(k // block, block, n),
            rhs_scale.view(torch.uint8),
        ),
    )
    source = functional_launch(
        lambda: source_module.cutile_block_scaled_matmul(
            lhs, lhs_scale, rhs, rhs_scale
        )
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=1.0, rtol=2e-2),
        cuda_graph=True,
    )


def dense_gemm(context: Context) -> PreparedComparison:
    m, k, n = 4096, 4096, 14336
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    _, generated = compile_single(
        context,
        gemm,
        (a, b),
        constexprs={"ACTIVATION": Activation.NONE},
        generated_candidate_filter=lambda candidate: (
            candidate_parameter_value(
                candidate, TRITON_PARAMETER_OWNERSHIP_M
            )
            in (64, 128)
            and candidate_parameter_value(
                candidate, TRITON_PARAMETER_OWNERSHIP_N
            )
            in (64, 128)
            and candidate_parameter_value(candidate, TRITON_PARAMETER_REDUCTION)
            in (32, 64)
            and candidate_parameter_value(
                candidate, TRITON_PARAMETER_TRAVERSAL_GROUP
            )
            == 8
        ),
    )
    source_module = official_source(
        context,
        "source/cutile/cutile-python/gemm/dense/MatMul_runtime.py",
        "intent_v2_cutile_dense_gemm",
    )
    source = functional_launch(lambda: source_module.cutile_matmul(a, b))
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=1e-2, rtol=1e-2),
        cuda_graph=True,
    )


def tilegym_dense_gemm(context: Context) -> PreparedComparison:
    m, k, n = 8192, 4096, 11008
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
    _, generated = compile_single(
        context,
        bf16_gemm,
        (a, b),
    )
    source_module = tilegym_source(
        context,
        "source/cutile/tilegym/gemm/dense/matmul.py",
        "tilegym_dense_gemm",
    )
    source = functional_launch(
        lambda: source_module.matmul(
            a,
            b,
            trans_a=False,
            trans_b=False,
            static_persistent=True,
        )
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=5e-2, rtol=2e-2),
        cuda_graph=True,
    )


def batched_gemm(context: Context) -> PreparedComparison:
    batch, m, k, n = 32, 512, 1024, 512
    a = torch.randn((batch, m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((batch, k, n), device="cuda", dtype=torch.bfloat16)
    _, generated = compile_single(context, batched_gemm_nn, (a, b))
    source_module = tilegym_source(
        context,
        "source/cutile/tilegym/gemm/batched/bmm.py",
        "batched_gemm",
    )
    source = functional_launch(
        lambda: source_module.bmm(
            a,
            b,
            transpose_a=False,
            transpose_b=False,
            static_persistent=True,
        )
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=5e-2, rtol=2e-2),
        cuda_graph=True,
    )


def grouped_gemm(context: Context) -> PreparedComparison:
    rows = (256, 512, 1024, 2048)
    boundaries = (0, 256, 768, 1792, 3840)
    hidden = output = 4096
    group_x = tuple(
        torch.randn((row, hidden), device="cuda", dtype=torch.bfloat16)
        for row in rows
    )
    group_weight = tuple(
        torch.randn((hidden, output), device="cuda", dtype=torch.bfloat16)
        for _ in rows
    )
    x = torch.cat(group_x, dim=0)
    weight = torch.stack(group_weight, dim=0)
    offsets = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
    _, generated = compile_single(
        context, ragged_grouped_gemm_bf16, (x, offsets, weight)
    )
    generated_tensor = generated.outputs()
    generated = PreparedLaunch(
        generated.launch,
        lambda: tuple(
            generated_tensor[begin:end]
            for begin, end in zip(boundaries[:-1], boundaries[1:])
        ),
    )
    source_module = tilegym_source(
        context,
        "source/cutile/tilegym/gemm/grouped/group_gemm.py",
        "grouped_gemm",
    )
    group_x_list = list(group_x)
    group_weight_list = list(group_weight)
    source = functional_launch(
        lambda: tuple(
            source_module.group_gemm(
                group_x_list,
                group_weight_list,
                static_persistent=True,
                transpose_b=False,
            )
        )
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=5e-2, rtol=2e-2),
        cuda_graph=False,
    )


CASES = {
    "block_scaled_gemm": block_scaled_gemm,
    "dense_gemm": dense_gemm,
    "tilegym_dense_gemm": tilegym_dense_gemm,
    "batched_gemm": batched_gemm,
    "grouped_gemm": grouped_gemm,
}
