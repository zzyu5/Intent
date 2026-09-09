import torch
from kernels.normalization.rms_norm import weighted_rms_norm
from ...model import Tolerance
from .common import prepare_comparison


def rms_norm(context):
    x = torch.randn((8192, 4096), dtype=torch.float32)
    weight = torch.randn((4096,), dtype=torch.float32)
    return prepare_comparison(context, weighted_rms_norm, (x, weight, 1.0 / 4096, 1e-6),
        "source/mojo/intentdsl/normalization/rms_norm/rms_norm_runtime.py", Tolerance(5e-5),
        "Source 使用 SIMD lane-striped sum，generated 保留 source-order-preserving reduction。")


CASES = {"weighted_rms_norm": rms_norm}
