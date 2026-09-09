import torch
from kernels.pointwise.batched_affine import batched_row_affine
from ...model import Tolerance
from .common import prepare_comparison


def affine(context):
    shape = (17, 257, 4093)
    x = torch.randn(shape, dtype=torch.float32)
    scale = torch.randn(shape[:2], dtype=torch.float32)
    bias = torch.randn(shape[:2], dtype=torch.float32)
    return prepare_comparison(context, batched_row_affine, (x, scale, bias),
        "source/mojo/intentdsl/pointwise/batched_affine/batched_affine_runtime.py", Tolerance(2e-6))


CASES = {"batched_row_affine": affine}
