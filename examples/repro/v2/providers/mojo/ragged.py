import math

import torch

from kernels.ragged.grouped_gemm import GROUPS
from kernels.ragged.grouped_gemm import K as GROUPED_K
from kernels.ragged.grouped_gemm import N as GROUPED_N
from kernels.ragged.grouped_gemm import ROWS as GROUPED_ROWS
from kernels.ragged.grouped_gemm import ragged_grouped_gemm
from kernels.ragged.jagged_mean import jagged_mean
from kernels.ragged.nested_pool import DOCUMENTS
from kernels.ragged.nested_pool import FEATURES as NESTED_FEATURES
from kernels.ragged.nested_pool import SENTENCES
from kernels.ragged.nested_pool import TOKENS as NESTED_TOKENS
from kernels.ragged.nested_pool import nested_jagged_mean_pool
from ...model import Tolerance
from .common import prepare_host_comparison


def jagged_mean_case(context):
    lengths = torch.arange(512, dtype=torch.int32) % 128 + 1
    offsets = torch.empty(513, dtype=torch.int32)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(lengths, dim=0)
    values = torch.randn((33024, 128), dtype=torch.float32)
    return prepare_host_comparison(context, jagged_mean, (values, offsets), "jagged_mean",
                                   Tolerance(atol=2e-5, rtol=1e-5))


def grouped_gemm(context):
    x = torch.randn((GROUPED_ROWS, GROUPED_K), dtype=torch.float16)
    x /= math.sqrt(GROUPED_K)
    offsets = torch.tensor(
        [
            0,
            GROUPED_ROWS // 32,
            GROUPED_ROWS // 16,
            GROUPED_ROWS // 8,
            GROUPED_ROWS // 4,
            GROUPED_ROWS // 2,
            3 * GROUPED_ROWS // 4,
            7 * GROUPED_ROWS // 8,
            GROUPED_ROWS,
        ],
        dtype=torch.int32,
    )
    weight = torch.randn(
        (GROUPS, GROUPED_K, GROUPED_N), dtype=torch.float16
    )
    return prepare_host_comparison(
        context,
        ragged_grouped_gemm,
        (x, offsets, weight),
        "ragged_grouped_gemm",
        Tolerance(atol=5e-2),
    )


def nested_pool(context):
    document_lengths = torch.tensor(
        [8 if index % 2 == 0 else 24 for index in range(DOCUMENTS)],
        dtype=torch.int32,
    )
    sentence_lengths = torch.tensor(
        [8 if index % 2 == 0 else 24 for index in range(SENTENCES)],
        dtype=torch.int32,
    )
    document_offsets = torch.zeros(DOCUMENTS + 1, dtype=torch.int32)
    sentence_offsets = torch.zeros(SENTENCES + 1, dtype=torch.int32)
    document_offsets[1:] = document_lengths.cumsum(dim=0)
    sentence_offsets[1:] = sentence_lengths.cumsum(dim=0)
    values = torch.randn((NESTED_TOKENS, NESTED_FEATURES), dtype=torch.float32)
    return prepare_host_comparison(
        context,
        nested_jagged_mean_pool,
        (document_offsets, sentence_offsets, values),
        "nested_jagged_mean_pool",
        (Tolerance(atol=2e-5), Tolerance(atol=2e-5)),
    )


CASES = {
    "jagged_mean": jagged_mean_case,
    "ragged_grouped_gemm": grouped_gemm,
    "nested_jagged_mean_pool": nested_pool,
}
