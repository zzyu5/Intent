import torch

from kernels.sorting.bitonic import bitonic_sort_rows
from kernels.sampling.top_k import insertion_top_k
from ...model import Tolerance
from .common import prepare_host_comparison


def bitonic_sort(context):
    values = torch.randn((4096, 256), dtype=torch.float32)
    return prepare_host_comparison(context, bitonic_sort_rows, (values,), "bitonic_sort",
                                   Tolerance(atol=0.0))


def top_k(context):
    logits = torch.randn((1024, 4093), dtype=torch.float32)
    return prepare_host_comparison(context, insertion_top_k, (logits,), "insertion_top_k",
                                   (Tolerance(atol=0.0), Tolerance(atol=0.0)))


CASES = {"bitonic_sort": bitonic_sort, "insertion_top_k": top_k}
