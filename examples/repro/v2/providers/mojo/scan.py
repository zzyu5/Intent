import torch

from kernels.compaction.nonzero import compact_nonzero_rows
from kernels.sampling.nucleus import sorted_nucleus_cutoff
from kernels.streaming.ordered_prefix import ordered_product_prefix, row_cumsum_f32
from ...model import Tolerance
from .common import prepare_host_comparison


def cumsum(context):
    x = torch.randn((8192, 8192), dtype=torch.float32)
    return prepare_host_comparison(context, row_cumsum_f32, (x,), "cumsum",
                                   Tolerance(atol=2e-3, rtol=1e-5))


def ordered_prefix(context):
    x = torch.randn((512, 17, 31), dtype=torch.float32) * 0.01
    return prepare_host_comparison(context, ordered_product_prefix, (x,), "ordered_prefix",
                                   Tolerance(atol=2e-5))


def nonzero(context):
    values = torch.randn((64, 4096), dtype=torch.float32)
    values[torch.rand_like(values) < 0.7] = 0.0
    return prepare_host_comparison(context, compact_nonzero_rows, (values,), "compact_nonzero",
                                   (Tolerance(atol=0.0), Tolerance(atol=0.0)))


def nucleus(context):
    ranks = torch.arange(4093, dtype=torch.float32)
    probabilities = torch.softmax(-ranks / 256.0, dim=0)[None, :].expand(1024, -1).contiguous()
    return prepare_host_comparison(context, sorted_nucleus_cutoff, (probabilities, 0.9), "nucleus",
                                   (Tolerance(atol=2e-4), Tolerance(atol=0.0)))


CASES = {"flaggems_cumsum": cumsum, "ordered_prefix": ordered_prefix,
         "compact_nonzero": nonzero, "sorted_nucleus_cutoff": nucleus}
