import math
from dataclasses import replace
import intent
import torch
from kernels.contraction.block_scaled import block_scaled_matmul
from kernels.contraction.block_scaled import deepgemm_fp8_2xacc, scaled_fp8_matmul, scaled_fp8_splitk_matmul
from kernels.contraction.block_sparse import block_sparse_matmul
from kernels.contraction.gemm import Activation, gemm_f32, bf16_gemm, gemm as half_gemm_definition
from kernels.contraction.gemm import quantized_gemm as quantized_gemm_definition
from kernels.contraction.batched_gemm import batched_gemm_nn
from kernels.contraction.batched_gemm import batched_gemm_nt as batched_gemm_nt_definition
from kernels.contraction.batched_gemm import batched_gemm_tn as batched_gemm_tn_definition
from kernels.contraction.batched_gemm import batched_gemm_tt as batched_gemm_tt_definition
from kernels.contraction.dual_gemm import gated_dual_gemm
from kernels.contraction.qkv import fused_qkv_projection
from kernels.contraction.sparse_2to4 import sparse_2to4_gemm
from kernels.contraction.weight_only_int4 import fp8_e4m3_matmul
from kernels.contraction.weight_only_int4 import fp8_e5m2_matmul, weight_only_int4_matmul
from kernels.contraction.weight_only_int4 import w4a8_packed_matmul, bitnet_int2_matmul, dequant_bf16_fp4_matmul
from kernels.contraction.mla import mla_head_projection as mla_head_projection_definition
from kernels.contraction.vector import vector_dot as vector_dot_definition
from kernels.contraction.vector import matrix_vector as matrix_vector_definition
from kernels.contraction.vector import vector_matrix as vector_matrix_definition
from kernels.contraction.vector import vector_outer as vector_outer_definition
from experiments._common.model import Tolerance, IntegerTolerance, SimilarityTolerance
from experiments._common.model import PreparedComparison, PreparedLaunch
from experiments._common.measurement import report_stage
from experiments._common.loading import load_module
from .common import configure_cpu_budget, prepare_comparison, prepare_host_comparison


def gemm(context):
    a = torch.randn((1024, 1024), dtype=torch.float32)
    b = torch.randn((1024, 1024), dtype=torch.float32)
    return prepare_comparison(context, gemm_f32, (a, b),
        "experiments/cpu/baselines/mojo/modular/contraction/gemm/gemm_runtime.py", Tolerance(2e-4, 1e-5),
        "Source 调用安装的 Modular/MAX CPU matmul，未缓存输入 packing。")


def vector_dot(context):
    lhs = torch.randn((1048576,), dtype=torch.float32)
    rhs = torch.randn_like(lhs)
    return prepare_host_comparison(context, vector_dot_definition, (lhs, rhs),
                                   "vector_dot", Tolerance(atol=2e-3, rtol=1e-5))


def matrix_vector(context):
    matrix = torch.randn((1024, 1024), dtype=torch.float32)
    vector = torch.randn((1024,), dtype=torch.float32)
    return prepare_host_comparison(context, matrix_vector_definition, (matrix, vector),
                                   "matmul", Tolerance(atol=2e-4, rtol=1e-5))


def vector_matrix(context):
    vector = torch.randn((1024,), dtype=torch.float32)
    matrix = torch.randn((1024, 1024), dtype=torch.float32)
    return prepare_host_comparison(context, vector_matrix_definition, (vector, matrix),
                                   "matmul", Tolerance(atol=2e-4, rtol=1e-5))


def vector_outer(context):
    lhs = torch.randn((1024,), dtype=torch.float32)
    rhs = torch.randn_like(lhs)
    return prepare_host_comparison(context, vector_outer_definition, (lhs, rhs),
                                   "vector_outer", Tolerance(atol=0.0))


def half_gemm(context):
    configure_cpu_budget()
    a = torch.randn((4096, 4096), dtype=torch.float16)
    b = torch.randn((4096, 14336), dtype=torch.float16)
    return prepare_host_comparison(context, half_gemm_definition, (a, b), "matmul", Tolerance(1e-2, 1e-2),
                                   constexprs={"ACTIVATION": Activation.NONE})


def bfloat_gemm(context):
    configure_cpu_budget()
    a = torch.randn((8192, 4096), dtype=torch.bfloat16)
    b = torch.randn((4096, 11008), dtype=torch.bfloat16)
    return prepare_host_comparison(context, bf16_gemm, (a, b), "matmul", Tolerance(5e-2, 2e-2))


def batched_gemm(context):
    configure_cpu_budget()
    a = torch.randn((32, 512, 1024), dtype=torch.bfloat16)
    b = torch.randn((32, 1024, 512), dtype=torch.bfloat16)
    return prepare_host_comparison(context, batched_gemm_nn, (a, b), "matmul", Tolerance(5e-2, 2e-2))


def batched_gemm_tn(context):
    configure_cpu_budget()
    logical_a = torch.randn((32, 512, 1024), dtype=torch.bfloat16)
    logical_a /= math.sqrt(1024)
    logical_b = torch.randn((32, 1024, 512), dtype=torch.bfloat16)
    a = logical_a.transpose(-1, -2).contiguous()
    return prepare_host_comparison(
        context,
        batched_gemm_tn_definition,
        (a, logical_b),
        "batched_gemm_tn",
        Tolerance(atol=5e-2),
    )


def batched_gemm_nt(context):
    configure_cpu_budget()
    logical_a = torch.randn((32, 512, 1024), dtype=torch.bfloat16)
    logical_a /= math.sqrt(1024)
    logical_b = torch.randn((32, 1024, 512), dtype=torch.bfloat16)
    b = logical_b.transpose(-1, -2).contiguous()
    return prepare_host_comparison(
        context,
        batched_gemm_nt_definition,
        (logical_a, b),
        "batched_gemm_nt",
        Tolerance(atol=5e-2),
    )


def batched_gemm_tt(context):
    configure_cpu_budget()
    logical_a = torch.randn((32, 512, 1024), dtype=torch.bfloat16)
    logical_a /= math.sqrt(1024)
    logical_b = torch.randn((32, 1024, 512), dtype=torch.bfloat16)
    a = logical_a.transpose(-1, -2).contiguous()
    b = logical_b.transpose(-1, -2).contiguous()
    return prepare_host_comparison(
        context,
        batched_gemm_tt_definition,
        (a, b),
        "batched_gemm_tt",
        Tolerance(atol=5e-2),
    )


def mla_head_query_projection(context):
    configure_cpu_budget()
    source = torch.randn((1, 512, 8, 128), dtype=torch.float16) * 0.1
    weight = torch.randn((8, 512, 128), dtype=torch.float16) * 0.05
    return prepare_host_comparison(
        context,
        mla_head_projection_definition,
        (source, weight),
        "mla_head_projection",
        Tolerance(atol=3e-2),
    )


def mla_head_value_projection(context):
    configure_cpu_budget()
    source = torch.randn((1, 512, 8, 512), dtype=torch.float16) * 0.1
    weight = torch.randn((8, 128, 512), dtype=torch.float16) * 0.05
    return prepare_host_comparison(
        context,
        mla_head_projection_definition,
        (source, weight),
        "mla_head_projection",
        Tolerance(atol=3e-2),
    )


def fp8_gemm(context):
    lhs = torch.randn((4096, 4096), dtype=torch.float16).to(torch.float8_e4m3fn)
    rhs = torch.randn((14336, 4096), dtype=torch.float16).to(torch.float8_e4m3fn)
    return prepare_host_comparison(context, fp8_e4m3_matmul, (lhs, rhs), "fp8_matmul",
                                   Tolerance(atol=0.5, rtol=5e-2))


def fp8_e5m2_gemm(context):
    lhs = torch.randn((1024, 1024), dtype=torch.float16).to(torch.float8_e5m2)
    rhs = torch.randn((1024, 1024), dtype=torch.float16).to(torch.float8_e5m2)
    comparison = prepare_host_comparison(context, fp8_e5m2_matmul, (lhs, rhs), "fp8_matmul",
                                         SimilarityTolerance(max_error=1e-3))
    return replace(comparison, note=comparison.note + " 沿用 extended FP8 E5M2 原始 normalized similarity error <=1e-3；未改变既有 E4M3 case 的逐元素容差。")


def quantized_gemm(context):
    lhs = torch.randn((4096, 4096), dtype=torch.float16) / math.sqrt(4096)
    rhs = torch.randn((4096, 14336), dtype=torch.float16)
    bias = torch.randn((14336,), dtype=torch.float32) * 0.25
    residual = torch.randn((4096, 14336), dtype=torch.float16) * 0.25
    scale = torch.linspace(1.0 / 48.0, 1.0 / 24.0, 14336, dtype=torch.float32)
    comparison = prepare_host_comparison(
        context, quantized_gemm_definition, (lhs, rhs, bias, residual, scale), "quantized_gemm",
        IntegerTolerance(max_abs=2),
    )
    return replace(comparison, note=comparison.note + " 原 GEMM+ReLU+bias+residual+scale+clamp 输出 i8；沿用 extended max_int8_error<=2。")


def weight_only_int4(context):
    activation = torch.randn((512, 2048), dtype=torch.float16) * 0.05
    logical = torch.randint(-8, 8, (2048, 4096), dtype=torch.int32)
    packed = torch.zeros((256, 4096), dtype=torch.int32)
    for lane in range(8):
        packed |= (logical[lane::8] & 15) << (4 * lane)
    scales = 0.01 + 0.02 * torch.rand((32, 4096), dtype=torch.float16)
    return prepare_host_comparison(context, weight_only_int4_matmul, (activation, packed, scales),
                                   "weight_only_int4_matmul", Tolerance(atol=4e-2))


def w4a8_gemm(context):
    activation = torch.randint(-128, 128, (4096, 4096), dtype=torch.int8)
    packed = torch.randint(0, 256, (14336, 2048), dtype=torch.uint8)
    comparison = prepare_host_comparison(context, w4a8_packed_matmul, (activation, packed),
                                         "w4a8_packed_matmul", Tolerance(atol=0.0))
    return replace(comparison, note=comparison.note + " 原 W4A8 4096x4096x14336，输出[N,M]i32；reference 含 signed nibble 解码，以 f32 精确承载本输入域的整数乘加后转 i32，逐元素精确比较。")


def bitnet_int2(context):
    activation = torch.randint(-8, 8, (1, 4096), dtype=torch.int8)
    logical = torch.randint(0, 2, (4096, 4096), dtype=torch.int8).reshape(4096, 256, 4, 4)
    packed = torch.zeros((4096, 256, 4), dtype=torch.uint8)
    for lane in range(4):
        packed |= logical[:, :, lane, :].to(torch.uint8) << (lane * 2)
    comparison = prepare_host_comparison(context, bitnet_int2_matmul, (activation, packed.flatten(1)),
                                         "bitnet_int2_matmul", Tolerance(atol=0.0))
    return replace(comparison, note=comparison.note + " 原 M1/N4096/K4096、逻辑 weight 0..1、int8 目标 interleaved INT2 packing；reference 含解码，f32 精确整数乘加后转 i32，逐元素精确比较。")


def dequant_bf16_fp4(context):
    activation = torch.randn((4096, 4096), dtype=torch.bfloat16) * 0.125
    packed = torch.randint(0, 256, (4096, 2048), dtype=torch.uint8)
    return prepare_host_comparison(context, dequant_bf16_fp4_matmul, (activation, packed),
                                   "dequant_bf16_fp4_matmul", Tolerance(atol=1.0, rtol=2e-2))


def deepgemm_fp8(context):
    lhs_values = torch.randn((4096, 4096), dtype=torch.bfloat16).view(4096, 32, 128)
    lhs_max = lhs_values.abs().float().amax(dim=2).clamp_min(1e-4)
    lhs = (lhs_values * (448.0 / lhs_max[:, :, None])).to(torch.float8_e4m3fn)
    rhs_values = torch.randn((4096, 4096), dtype=torch.bfloat16).view(32, 128, 32, 128)
    rhs_max = rhs_values.abs().float().amax(dim=(1, 3), keepdim=True).clamp_min(1e-4)
    rhs = (rhs_values * (448.0 / rhs_max)).to(torch.float8_e4m3fn).view(4096, 32, 128)
    return prepare_host_comparison(context, deepgemm_fp8_2xacc,
                                   (lhs, rhs, lhs_max / 448.0, (rhs_max / 448.0).view(32, 32)),
                                   "deepgemm_fp8_2xacc", Tolerance(atol=1.0, rtol=2e-2))


def scaled_fp8(context):
    lhs = torch.randn((4096, 4096), dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    rhs = torch.randn((4096, 14336), dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    report_stage("generated_compilation")
    artifact = intent.compile(scaled_fp8_matmul, target=context.target, compiler=context.compiler,
                              tuning_config=context.tuning_config, options=context.compile_options)
    state = {}

    def launch():
        state["output"] = artifact.run(lhs, rhs, 1.0, 1.0)

    report_stage("adapter_preparation")
    return PreparedComparison(PreparedLaunch(launch, lambda: state["output"]), None, None,
        cuda_graph=False, status="run_only", device_type="cpu", cpu_host_timing=True,
        note="原作者普通 scalar-scaled FP8 GEMM，复用同文件 split-K production 的4096x4096x14336输入规模、FP8 dtype和scale1，输出f16；没有该entry的现成reference，仅验证完整调用可运行，含输出分配和同步。")


def scaled_fp8_splitk(context):
    lhs = torch.randn((4096, 4096), dtype=torch.bfloat16).to(torch.float8_e4m3fn).view(4096, 4, 4, 256)
    rhs = torch.randn((4096, 14336), dtype=torch.bfloat16).to(torch.float8_e4m3fn).view(4, 4, 256, 14336)
    report_stage("generated_compilation")
    artifact = intent.compile(scaled_fp8_splitk_matmul, target=context.target, compiler=context.compiler,
                              tuning_config=context.tuning_config, options=context.compile_options)
    runtime = load_module(context.project_root / "experiments/cpu/baselines/pytorch/cpu_runtime.py", "intent_cpu_reference")

    def side(function):
        output = torch.zeros((4096, 14336), dtype=torch.float16)
        return PreparedLaunch(lambda: function(lhs, rhs, output, 1.0, 1.0), lambda: output,
                              prepare=lambda: output.zero_())

    report_stage("adapter_preparation")
    return PreparedComparison(side(artifact.run), side(runtime.scaled_fp8_splitk_matmul),
        Tolerance(atol=5e-1, rtol=1e-2), cuda_graph=False, device_type="cpu", cpu_host_timing=True,
        note="原4096x4096x14336/IT4/split4/Kblock256 FP8 GEMM；每split独立f32累加、缩放并转f16，再对InOut输出f16相加；双方输出预分配，每次清零不计时；PyTorch CPU reference采用递增split顺序，保留原容差。")


def _block_quantize_mxfp8(x, block_size):
    dtype_max = torch.finfo(torch.float8_e4m3fn).max
    x_block = x.reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)
    scale = torch.max(x_block.abs(), dim=-1, keepdims=True)[0]
    scale = torch.clamp(scale / dtype_max, min=1e-12)
    scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    x_q = (x_block / scale).to(torch.float8_e4m3fn).reshape(x.shape)
    scale = scale.to(torch.float8_e8m0fnu).squeeze(-1)
    return x_q, scale


def mxfp8_gemm(context):
    configure_cpu_budget()
    m, k, n, block = 4096, 4096, 14336, 32
    lhs_rows, lhs_scale_rows = _block_quantize_mxfp8(
        torch.randn((m, k), dtype=torch.float32), block
    )
    rhs_rows, rhs_scale_rows = _block_quantize_mxfp8(
        torch.randn((n, k), dtype=torch.float32), block
    )
    lhs = lhs_rows.view(m, k // block, block)
    lhs_scale = lhs_scale_rows.view(torch.uint8)
    rhs = rhs_rows.T.view(k // block, block, n)
    rhs_scale = rhs_scale_rows.T.view(torch.uint8)
    return prepare_host_comparison(
        context,
        block_scaled_matmul,
        (lhs, lhs_scale, rhs, rhs_scale),
        "block_scaled_matmul",
        Tolerance(atol=1.0, rtol=2e-2),
    )


def block_sparse_gemm(context):
    configure_cpu_budget()
    dimension = 4096
    lhs = torch.randn((dimension, dimension), dtype=torch.float16)
    rhs = torch.randn_like(lhs)
    mask = (torch.rand((32, 32, 128)) > 0.5).to(torch.uint8)
    return prepare_host_comparison(
        context,
        block_sparse_matmul,
        (lhs, rhs, mask),
        "block_sparse_matmul",
        Tolerance(atol=5e-2, rtol=2e-2),
    )


def sparse_2to4(context):
    configure_cpu_budget()
    m, n, k = 8192, 14336, 8192
    runtime = load_module(context.project_root / "experiments/cpu/baselines/pytorch/cpu_runtime.py", "intent_cpu_sparse_inputs")
    compressed, metadata = runtime.make_sparse_2to4_inputs(m, k)
    rhs = torch.randn((k, n), dtype=torch.float16)
    comparison = prepare_host_comparison(
        context,
        sparse_2to4_gemm,
        (compressed, metadata, rhs),
        "sparse_2to4_gemm",
        Tolerance(atol=5.0e-2, rtol=2.0e-2),
    )
    return replace(comparison, note=comparison.note +
                   " CPU reference 计入逻辑稀疏输入的 dense 解码及 f32 GEMM；生成程序保留压缩非零遍历。")


def qkv_projection(context):
    configure_cpu_budget()
    tokens = hidden = projection = 4096
    x = torch.randn((tokens, hidden), dtype=torch.float16)
    weights = tuple(
        torch.randn((hidden, projection), dtype=torch.float16) for _ in range(3)
    )
    packed_weights = torch.stack(weights)

    return prepare_host_comparison(context, fused_qkv_projection, (x, packed_weights),
                                   "qkv_projection", Tolerance(atol=2e-2, rtol=1e-2))


def dual_gemm(context):
    configure_cpu_budget()
    x = torch.randn((2048, 4096), dtype=torch.float16)
    x /= math.sqrt(4096)
    gate_weight = torch.randn((4096, 4096), dtype=torch.float16)
    value_weight = torch.randn_like(gate_weight)
    return prepare_host_comparison(
        context,
        gated_dual_gemm,
        (x, gate_weight, value_weight),
        "gated_dual_gemm",
        Tolerance(atol=5e-2),
    )


CASES = {"dense_gemm_f32": gemm, "dense_gemm": half_gemm, "tilegym_dense_gemm": bfloat_gemm,
         "batched_gemm": batched_gemm, "batched_gemm_tn": batched_gemm_tn,
         "batched_gemm_nt": batched_gemm_nt, "batched_gemm_tt": batched_gemm_tt,
         "mla_head_query_projection": mla_head_query_projection,
         "mla_head_value_projection": mla_head_value_projection,
         "qkv_projection": qkv_projection,
         "gated_dual_gemm": dual_gemm, "fp8_gemm": fp8_gemm,
         "fp8_e5m2_gemm": fp8_e5m2_gemm, "quantized_gemm": quantized_gemm,
         "weight_only_int4": weight_only_int4,
         "w4a8_gemm": w4a8_gemm, "bitnet_int2_decode": bitnet_int2,
         "dequant_bf16_fp4": dequant_bf16_fp4,
         "deepgemm_fp8_2xacc": deepgemm_fp8, "scaled_fp8_gemm": scaled_fp8,
         "scaled_fp8_splitk_gemm": scaled_fp8_splitk,
         "vector_dot": vector_dot, "matrix_vector": matrix_vector,
         "vector_matrix": vector_matrix, "vector_outer": vector_outer,
         "mxfp8_gemm": mxfp8_gemm, "block_sparse_gemm": block_sparse_gemm,
         "sparse_2to4_gemm": sparse_2to4}
