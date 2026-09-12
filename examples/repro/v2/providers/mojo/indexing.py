import torch

from kernels.backward.embedding import embedding_forward_lookup
from kernels.backward.embedding import embedding_forward_lookup_bf16
from kernels.indexing.relations import index_select_rows
from kernels.indexing.relations import scalar_table_lookup
from ...model import Tolerance
from .common import prepare_host_comparison


def embedding(context):
    table = torch.randn((32768, 4096), dtype=torch.bfloat16)
    indices = torch.randint(0, 32768, (8, 2048), dtype=torch.int64).reshape(-1)
    return prepare_host_comparison(context, embedding_forward_lookup_bf16,
        (table, indices), "index_select", Tolerance(atol=0.0))


def embedding_f32(context):
    table = torch.randn((8192, 1021), dtype=torch.float32)
    indices = torch.randint(0, 8192, (8192,), dtype=torch.int32)
    return prepare_host_comparison(context, embedding_forward_lookup,
        (table, indices), "embedding_forward_lookup", Tolerance(atol=0.0))


def index_select(context):
    source = torch.randn((65536, 4096), dtype=torch.float16)
    indices = torch.arange(0, 32768 * 2, 2, dtype=torch.int64)
    return prepare_host_comparison(context, index_select_rows,
        (source, indices), "index_select", Tolerance(atol=0.0))


def scalar_lookup(context):
    labels = torch.randint(0, 65536, (65536,), dtype=torch.int32)
    table = torch.randn((65536, 128), dtype=torch.float32)
    return prepare_host_comparison(context, scalar_table_lookup,
        (labels, table), "scalar_table_lookup", Tolerance(atol=0.0))


CASES = {"embedding_lookup": embedding, "embedding_lookup_f32": embedding_f32,
         "index_select": index_select,
         "scalar_table_lookup": scalar_lookup}
