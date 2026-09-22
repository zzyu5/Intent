import torch

from kernels.layout.transpose import matrix_transpose

from experiments._common.model import Tolerance
from .common import configure_cpu_budget, prepare_host_comparison


def transpose(context):
    configure_cpu_budget()
    x = torch.randn((4093, 8191), dtype=torch.float16)
    return prepare_host_comparison(context, matrix_transpose, (x,), "transpose", Tolerance(0.0))


CASES = {"flaggems_transpose_copy": transpose}
