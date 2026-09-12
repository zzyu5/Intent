import torch
from kernels.pointwise.batched_affine import batched_row_affine
from kernels.pointwise.while_loop import integer_log2_floor
from ...model import Tolerance
from .common import prepare_comparison, prepare_host_comparison


def affine(context):
    shape = (17, 257, 4093)
    x = torch.randn(shape, dtype=torch.float32)
    scale = torch.randn(shape[:2], dtype=torch.float32)
    bias = torch.randn(shape[:2], dtype=torch.float32)
    return prepare_comparison(context, batched_row_affine, (x, scale, bias),
        "source/mojo/modular/pointwise/batched_affine/batched_affine_runtime.py", Tolerance(2e-6),
        "Source 调用安装的 Modular/MAX CPU elementwise。")


def integer_log2(context):
    values = torch.randint(1, 1 << 30, (262144,), dtype=torch.int32)
    return prepare_host_comparison(context, integer_log2_floor, (values,),
        "integer_log2_floor", Tolerance(atol=0.0))


CASES = {"batched_row_affine": affine, "integer_log2_floor": integer_log2}
