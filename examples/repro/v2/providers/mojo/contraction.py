import math
import torch
from kernels.contraction.gemm import Activation, gemm_f32, bf16_gemm, gemm as half_gemm_definition
from kernels.contraction.batched_gemm import batched_gemm_nn
from kernels.contraction.dual_gemm import gated_dual_gemm
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
         "batched_gemm": batched_gemm, "gated_dual_gemm": dual_gemm}
