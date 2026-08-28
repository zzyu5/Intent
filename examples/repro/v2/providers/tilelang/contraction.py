from __future__ import annotations

import torch
import intent

from repro.common.support import prepare_kernel_call

from kernels.contraction.block_sparse import block_sparse_matmul
from kernels.contraction.block_scaled import deepgemm_fp8_2xacc
from kernels.contraction.gemm import Activation
from kernels.contraction.gemm import gemm
from kernels.contraction.sparse_2to4 import sparse_2to4_gemm
from kernels.contraction.weight_only_int4 import fp8_e4m3_matmul
from kernels.contraction.weight_only_int4 import bitnet_int2_matmul
from kernels.contraction.weight_only_int4 import dequant_bf16_fp4_matmul
from kernels.contraction.weight_only_int4 import w4a8_packed_matmul
from kernels.ragged.grouped_gemm import ragged_grouped_gemm
from kernels.ragged.grouped_gemm import ragged_grouped_gemm_backward_weight

from ...measurement import compile_single
from ...measurement import candidate_parameter_value
from ...measurement import TRITON_PARAMETER_OWNERSHIP_M
from ...measurement import TRITON_PARAMETER_OWNERSHIP_N
from ...measurement import TRITON_PARAMETER_PROVIDER_STAGES
from ...measurement import TRITON_PARAMETER_PROVIDER_THREADS
from ...measurement import TRITON_PARAMETER_REDUCTION
from ...measurement import TRITON_PARAMETER_TRAVERSAL_GROUP
from ...measurement import functional_launch
from ...model import Context
from ...model import PreparedComparison
from ...model import PreparedLaunch
from ...model import Tolerance
from .common import runtime_module
from .common import source_from_runtime


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
            == 128
            and candidate_parameter_value(
                candidate, TRITON_PARAMETER_OWNERSHIP_N
            )
            == 128
            and candidate_parameter_value(candidate, TRITON_PARAMETER_REDUCTION)
            == 32
            and candidate_parameter_value(
                candidate, TRITON_PARAMETER_PROVIDER_STAGES
            )
            == 3
            and candidate_parameter_value(
                candidate, TRITON_PARAMETER_PROVIDER_THREADS
            )
            == 128
            and candidate_parameter_value(
                candidate, TRITON_PARAMETER_TRAVERSAL_GROUP
            )
            == 1
        ),
    )
    _, source_module = source_from_runtime(
        context,
        "source/tilelang/tilelang/gemm/dense/example_gemm_runtime.py",
        "intent_v2_tilelang_dense_gemm",
    )
    source_kernel = source_module.matmul.compile(
        M=m,
        N=n,
        K=k,
        block_M=128,
        block_N=128,
        block_K=32,
    )
    source = functional_launch(lambda: source_kernel(a, b))
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=1e-2, rtol=1e-2),
        cuda_graph=True,
    )


def w4a8_gemm(context: Context) -> PreparedComparison:
    m, k, n = 4096, 4096, 14336
    activation = torch.randint(
        -128, 128, (m, k), device="cuda", dtype=torch.int8
    )
    packed = torch.randint(
        0, 256, (n, k // 2), device="cuda", dtype=torch.uint8
    )
    _, generated = compile_single(
        context, w4a8_packed_matmul, (activation, packed)
    )
    _, source_module = source_from_runtime(
        context,
        "source/tilelang/tilelang/gemm/dequantize_w4a8/example_dequant_gemm_w4a8_runtime.py",
        "intent_v2_tilelang_w4a8",
    )
    source_kernel = source_module.matmul_int8xint4(
        m,
        n,
        k,
        source_module.T.int8,
        source_module.T.int32,
        source_module.T.int32,
        num_bits=4,
        block_M=128,
        block_N=128,
        block_K=128,
        num_stages=2,
        threads=256,
    )
    source = functional_launch(lambda: source_kernel(activation, packed))
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=0.0),
        cuda_graph=True,
    )


def bitnet_int2(context: Context) -> PreparedComparison:
    rows, output_dimension, reduction = 1, 4096, 4096
    activation = torch.randint(
        -8,
        8,
        (rows, reduction),
        device="cuda",
        dtype=torch.int8,
    )
    support = runtime_module(
        context,
        "source/tilelang/tilelang/support/runtime.py",
        "intent_v2_tilelang_runtime_support",
    )
    source_module = support.load_source(
        context.project_root
        / "source/tilelang/tilelang/gemm/bitnet_int2_decode/tilelang_bitnet_158_int8xint2_decode.py",
        "intent_v2_tilelang_bitnet_int2",
    )
    logical_weight = torch.randint(
        0,
        2,
        (output_dimension, reduction),
        dtype=torch.int8,
    ).numpy()
    packed_numpy = source_module.general_compress(
        logical_weight,
        source_bits=2,
    )
    packed_numpy = source_module.interleave_weight(
        packed_numpy,
        2,
        target_dtype=source_module.T.int8,
    )
    packed_i8 = torch.from_numpy(packed_numpy).to(device="cuda")
    packed_u8 = packed_i8.view(torch.uint8)
    _, generated = compile_single(
        context,
        bitnet_int2_matmul,
        (activation, packed_u8),
    )
    source_program = source_module.bitnet_158_int8xint2_decode(
        rows,
        output_dimension,
        reduction,
        source_module.T.int8,
        source_module.T.int32,
        source_module.T.int32,
    )
    source_kernel = source_module.tilelang.compile(source_program)
    source_output = torch.empty(
        (rows, output_dimension),
        device="cuda",
        dtype=torch.int32,
    )

    def source_call():
        source_kernel(activation, packed_i8, source_output)
        return source_output

    source = functional_launch(source_call)
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=0.0),
        cuda_graph=True,
    )


def fp8_gemm(context: Context) -> PreparedComparison:
    m, k, n = 4096, 4096, 14336
    _, source_module = source_from_runtime(
        context,
        "source/tilelang/tilelang/gemm/fp8/example_tilelang_gemm_fp8_runtime.py",
        "intent_v2_tilelang_fp8_gemm",
    )
    dtype = source_module.determine_fp8_type()
    torch_dtype = source_module.T.dtype(dtype).as_torch()
    lhs = torch.randn((m, k), device="cuda", dtype=torch.float16).to(torch_dtype)
    rhs = torch.randn((n, k), device="cuda", dtype=torch.float16).to(torch_dtype)
    _, generated = compile_single(
        context,
        fp8_e4m3_matmul,
        (lhs, rhs),
        generated_candidate_filter=lambda candidate: (
            candidate_parameter_value(
                candidate, TRITON_PARAMETER_OWNERSHIP_M
            )
            == 128
            and candidate_parameter_value(
                candidate, TRITON_PARAMETER_OWNERSHIP_N
            )
            == 128
            and candidate_parameter_value(candidate, TRITON_PARAMETER_REDUCTION)
            == 64
            and candidate_parameter_value(
                candidate, TRITON_PARAMETER_PROVIDER_STAGES
            )
            == 3
            and candidate_parameter_value(
                candidate, TRITON_PARAMETER_PROVIDER_THREADS
            )
            == 128
            and candidate_parameter_value(
                candidate, TRITON_PARAMETER_TRAVERSAL_GROUP
            )
            == 1
        ),
    )
    source_kernel = source_module.matmul.compile(
        M=m,
        N=n,
        K=k,
        block_M=128,
        block_N=128,
        block_K=64,
        dtype=dtype,
    )
    source = functional_launch(lambda: source_kernel(lhs, rhs))
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=0.5, rtol=5e-2),
        cuda_graph=True,
    )


def grouped_gemm(context: Context) -> PreparedComparison:
    rows = (256, 512, 1024, 2048)
    hidden = output = 4096
    runtime = runtime_module(
        context,
        "source/tilelang/tilelang/gemm/grouped/example_grouped_gemm_fwd_runtime.py",
        "intent_v2_tilelang_grouped_gemm_runtime",
    )
    a, b, sizes, offsets, padded_offsets = runtime.source.construct_inputs(
        rows,
        hidden,
        output,
        False,
        64,
        torch.device("cuda"),
        torch.float16,
    )
    group_offsets = torch.cat(
        (offsets, torch.tensor((sum(rows),), device="cuda", dtype=torch.int32))
    )
    _, generated = compile_single(
        context, ragged_grouped_gemm, (a, group_offsets, b)
    )
    source = functional_launch(
        lambda: runtime.source.grouped_gemm(
            a,
            b,
            sizes,
            offsets,
            padded_offsets,
            rows,
            64,
            128,
            64,
            False,
            2,
            256,
        )
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=2e-2, rtol=1e-2),
        cuda_graph=False,
    )


def sparse_2to4(context: Context) -> PreparedComparison:
    m, n, k = 8192, 14336, 8192
    runtime = runtime_module(
        context,
        "source/tilelang/tilelang/gemm/sparse_2to4/example_gemm_sp_runtime.py",
        "intent_v2_tilelang_sparse_2to4_runtime",
    )
    dense = runtime.source.randn_semi_sparse(
        m, k, device="cuda", dtype=torch.float16
    )
    rhs = torch.randn((k, n), device="cuda", dtype=torch.float16)
    compressed, metadata = runtime.sparse_utils.torch_compress(
        dense, meta_dtype=torch.int16
    )
    _, generated = compile_single(
        context, sparse_2to4_gemm, (compressed, metadata, rhs)
    )
    source_kernel = runtime.source.matmul_sp_fp16(
        m,
        n,
        k,
        runtime.source.T.float,
        runtime.source.T.int16,
        128,
        128,
        64,
        2,
        128,
        runtime.source.T.GemmWarpPolicy.Square,
        True,
    )
    source = functional_launch(lambda: source_kernel(compressed, metadata, rhs))
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=5e-2, rtol=2e-2),
        cuda_graph=True,
    )


def block_sparse(context: Context) -> PreparedComparison:
    dimension = 4096
    lhs = torch.randn(
        (dimension, dimension), device="cuda", dtype=torch.float16
    )
    rhs = torch.randn_like(lhs)
    mask = torch.rand((32, 32, 128), device="cuda") > 0.5
    _, generated = compile_single(
        context, block_sparse_matmul, (lhs, rhs, mask.to(torch.uint8))
    )
    runtime = runtime_module(
        context,
        "source/tilelang/tilelang/gemm/block_sparse/example_blocksparse_gemm_runtime.py",
        "intent_v2_tilelang_block_sparse_runtime",
    )
    source_kernel = runtime.load_source().blocksparse_matmul
    source = functional_launch(
        lambda: source_kernel(
            lhs, rhs, mask, 128, 128, 32, 2, 128, True
        )
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=5e-2, rtol=2e-2),
        cuda_graph=False,
    )


def grouped_gemm_backward(context: Context) -> PreparedComparison:
    rows = (256, 512, 1024, 2048)
    hidden = output = 4096
    total = sum(rows)
    left = torch.randn((total, hidden), device="cuda", dtype=torch.float16)
    right = torch.randn((total, output), device="cuda", dtype=torch.float16)
    sizes = torch.tensor(rows, device="cuda", dtype=torch.int32)
    offsets = torch.tensor((0, 256, 768, 1792), device="cuda", dtype=torch.int32)
    group_offsets = torch.cat(
        (offsets, torch.tensor((total,), device="cuda", dtype=torch.int32))
    )
    grad_weight = torch.empty(
        (len(rows), hidden, output), device="cuda", dtype=torch.float16
    )
    artifact = intent.compile(
        ragged_grouped_gemm_backward_weight,
        target=context.target,
        compiler=context.compiler,
    )
    generated = PreparedLaunch(
        launch=prepare_kernel_call(
            artifact, (left, right, group_offsets), grad_weight
        ),
        outputs=lambda: grad_weight,
    )
    runtime = runtime_module(
        context,
        "source/tilelang/tilelang/gemm/grouped_backward/example_grouped_gemm_bwd_runtime.py",
        "intent_v2_tilelang_grouped_gemm_backward_runtime",
    )
    source_kernel = runtime.load_source().grouped_gemm_bwd
    source = functional_launch(
        lambda: source_kernel(
            left,
            right,
            sizes,
            offsets,
            64,
            128,
            64,
            num_stages=2,
            threads=256,
        )
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=5e-2, rtol=2e-2),
        cuda_graph=False,
    )


def deepgemm_fp8(context: Context) -> PreparedComparison:
    m = n = k = 4096
    support = runtime_module(
        context,
        "source/tilelang/tilelang/support/runtime.py",
        "intent_v2_tilelang_runtime_support",
    )
    source_module = support.load_source(
        context.project_root
        / "source/tilelang/tilelang/gemm/fp8_2xacc/example_deepgemm_fp8_2xAcc.py",
        "intent_v2_tilelang_deepgemm_fp8",
    )
    lhs, lhs_scale = source_module.per_token_cast_to_fp8(
        torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    )
    rhs, rhs_scale = source_module.per_block_cast_to_fp8(
        torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    )
    _, generated = compile_single(
        context,
        deepgemm_fp8_2xacc,
        (
            lhs.view(m, k // 128, 128),
            rhs.view(n, k // 128, 128),
            lhs_scale,
            rhs_scale,
        ),
    )
    source = functional_launch(
        lambda: source_module.tl_gemm(
            lhs,
            rhs,
            lhs_scale,
            rhs_scale,
            128,
            source_module.T.float8_e4m3fn,
            source_module.T.bfloat16,
            source_module.T.float32,
        )
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=1.0, rtol=2e-2),
        cuda_graph=False,
    )


def dequant_bf16_fp4(context: Context) -> PreparedComparison:
    m = n = k = 4096
    activation = (
        torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.125
    )
    packed_weight = torch.randint(
        0,
        256,
        (n, k // 2),
        device="cuda",
        dtype=torch.uint8,
    )
    _, generated = compile_single(
        context,
        dequant_bf16_fp4_matmul,
        (activation, packed_weight),
    )
    support = runtime_module(
        context,
        "source/tilelang/tilelang/support/runtime.py",
        "intent_v2_tilelang_runtime_support",
    )
    source_module = support.load_source(
        context.project_root
        / "source/tilelang/tilelang/gemm/dequant_bf16_fp4/example_dequant_gemm_bf16_fp4_hopper.py",
        "intent_v2_tilelang_dequant_bf16_fp4",
    )
    source_kernel = source_module.matmul(
        m,
        n,
        k,
        "bfloat16",
        "bfloat16",
        "float32",
        num_bits=4,
        fast_dequant=True,
        block_M=256,
        block_N=128,
        block_K=128,
        num_stages=2,
        threads=256,
        split=1,
    )
    source = functional_launch(lambda: source_kernel(activation, packed_weight))
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=1.0, rtol=2e-2),
        cuda_graph=False,
    )


CASES = {
    "dense_gemm": dense_gemm,
    "w4a8_gemm": w4a8_gemm,
    "fp8_gemm": fp8_gemm,
    "grouped_gemm": grouped_gemm,
    "sparse_2to4_gemm": sparse_2to4,
    "block_sparse_gemm": block_sparse,
    "grouped_gemm_backward": grouped_gemm_backward,
    "deepgemm_fp8_2xacc": deepgemm_fp8,
    "bitnet_int2_decode": bitnet_int2,
    "dequant_bf16_fp4": dequant_bf16_fp4,
}
