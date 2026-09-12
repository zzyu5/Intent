import math
import torch
from kernels.contraction.gemm import Activation, gemm_f32, bf16_gemm, gemm as half_gemm_definition
from kernels.contraction.batched_gemm import batched_gemm_nn
from kernels.contraction.batched_gemm import batched_gemm_nt as batched_gemm_nt_definition
from kernels.contraction.batched_gemm import batched_gemm_tn as batched_gemm_tn_definition
from kernels.contraction.batched_gemm import batched_gemm_tt as batched_gemm_tt_definition
from kernels.contraction.dual_gemm import gated_dual_gemm
from kernels.contraction.mla import mla_head_projection as mla_head_projection_definition
from ...model import Tolerance
from .common import configure_cpu_budget, prepare_comparison, prepare_host_comparison


def gemm(context):
    a = torch.randn((1024, 1024), dtype=torch.float32)
    b = torch.randn((1024, 1024), dtype=torch.float32)
    return prepare_comparison(context, gemm_f32, (a, b),
        "source/mojo/modular/contraction/gemm/gemm_runtime.py", Tolerance(2e-4, 1e-5),
        "Source 调用安装的 Modular/MAX CPU matmul，未缓存输入 packing。")


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
         "gated_dual_gemm": dual_gemm}
