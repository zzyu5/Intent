import torch
from kernels.contraction.gemm import gemm_f32
from ...model import Tolerance
from .common import prepare_comparison


def gemm(context):
    a = torch.randn((1024, 1024), dtype=torch.float32)
    b = torch.randn((1024, 1024), dtype=torch.float32)
    return prepare_comparison(context, gemm_f32, (a, b),
        "source/mojo/modular/contraction/gemm/gemm_runtime.py", Tolerance(2e-4, 1e-5),
        "Source 调用安装的 Modular/MAX CPU matmul，未缓存输入 packing。")


CASES = {"dense_gemm_f32": gemm}
