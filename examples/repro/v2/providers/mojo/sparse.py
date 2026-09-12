import torch

from kernels.sparse.csr_spmv import csr_spmv

from ...model import Tolerance
from .common import prepare_host_comparison


def csr(context):
    rows = columns = 32768
    nonzeros_per_row = 32
    nonzeros = rows * nonzeros_per_row
    row_offsets = torch.arange(rows + 1, dtype=torch.int32) * nonzeros_per_row
    column_indices = torch.randint(0, columns, (nonzeros,), dtype=torch.int32)
    values = torch.randn((nonzeros,), dtype=torch.float32) * 0.1
    vector = torch.randn((columns,), dtype=torch.float32)
    return prepare_host_comparison(context, csr_spmv,
        (row_offsets, column_indices, values, vector), "csr_spmv",
        Tolerance(atol=2.0e-5))


CASES = {"csr_spmv": csr}
