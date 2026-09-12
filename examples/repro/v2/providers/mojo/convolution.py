import torch

from kernels.convolution.direct import conv1d_same
from ...model import Tolerance
from .common import prepare_host_comparison


def conv1d(context):
    x = torch.randn((64, 16384), dtype=torch.float16)
    weight = torch.randn((5,), dtype=torch.float16)
    return prepare_host_comparison(context, conv1d_same, (x, weight), "conv1d_same",
                                   Tolerance(atol=2e-2, rtol=1e-2))


CASES = {"flaggems_conv1d": conv1d}
