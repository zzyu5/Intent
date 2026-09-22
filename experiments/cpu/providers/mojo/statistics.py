import torch

from kernels.statistics.histogram import histogram_256
from experiments._common.model import Tolerance
from .common import prepare_host_comparison


def histogram(context):
    samples = torch.randint(0, 256, (8 * 1024 * 1024,), dtype=torch.int32).to(torch.float32)
    return prepare_host_comparison(context, histogram_256, (samples,), "histogram",
                                   Tolerance(atol=0.0))


CASES = {"histogram": histogram}
