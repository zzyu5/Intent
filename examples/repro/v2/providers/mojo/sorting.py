import torch

from kernels.sorting.bitonic import bitonic_sort_rows
from ...model import Tolerance
from .common import prepare_host_comparison


def bitonic_sort(context):
    values = torch.randn((4096, 256), dtype=torch.float32)
    return prepare_host_comparison(context, bitonic_sort_rows, (values,), "bitonic_sort",
                                   Tolerance(atol=0.0))


CASES = {"bitonic_sort": bitonic_sort}
