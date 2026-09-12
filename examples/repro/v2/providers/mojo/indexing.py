import torch

from kernels.backward.embedding import embedding_forward_lookup
from kernels.backward.embedding import embedding_forward_lookup_bf16
from kernels.indexing.relations import (
    GQA_KEY_HEADS,
    GQA_QUERY_HEADS,
    GQA_TOKENS,
    OFFSET_FEATURES,
    OFFSET_ROWS,
    grouped_query_head_add,
    shifted_row_copy,
    roll_rows_forward,
)
from kernels.indexing.relations import index_select_rows
from kernels.indexing.relations import scalar_table_lookup
from kernels.pointwise.select import COLUMNS as SELECT_COLUMNS
from kernels.pointwise.select import ROWS as SELECT_ROWS
from kernels.pointwise.select import alternating_signed_indices
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


def shifted_row(context):
    x = torch.randn((OFFSET_ROWS, OFFSET_FEATURES), dtype=torch.float32)
    return prepare_host_comparison(
        context,
        shifted_row_copy,
        (x,),
        "shifted_row_copy",
        Tolerance(atol=0.0),
    )


def rolled_rows(context):
    x = torch.randn((8192, 4096), dtype=torch.float16).reshape(-1)
    return prepare_host_comparison(
        context,
        roll_rows_forward,
        (x,),
        "roll_rows_forward",
        Tolerance(atol=0.0),
        constexprs={"ROW_WIDTH": 4096},
    )


def grouped_query_heads(context):
    query = torch.randn((GQA_QUERY_HEADS, GQA_TOKENS), dtype=torch.float16)
    key = torch.randn((GQA_KEY_HEADS, GQA_TOKENS), dtype=torch.float16)
    return prepare_host_comparison(
        context,
        grouped_query_head_add,
        (query, key),
        "grouped_query_head_add",
        Tolerance(atol=0.0),
    )


def alternating_indices(context):
    shape_source = torch.empty(
        (SELECT_ROWS, SELECT_COLUMNS), dtype=torch.float32
    )
    return prepare_host_comparison(
        context,
        alternating_signed_indices,
        (shape_source,),
        "alternating_signed_indices",
        Tolerance(atol=0.0),
    )


CASES = {"embedding_lookup": embedding, "embedding_lookup_f32": embedding_f32,
         "index_select": index_select,
         "scalar_table_lookup": scalar_lookup,
         "shifted_row_copy": shifted_row,
         "roll_rows_forward": rolled_rows,
         "grouped_query_head_add": grouped_query_heads,
         "alternating_signed_indices": alternating_indices}
