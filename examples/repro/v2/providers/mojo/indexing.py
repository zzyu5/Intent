import torch
import intent

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
from kernels.indexing.relations import scaled_index_add_unique
from kernels.pointwise.select import COLUMNS as SELECT_COLUMNS
from kernels.pointwise.select import ROWS as SELECT_ROWS
from kernels.pointwise.select import alternating_signed_indices
from ...loading import load_module
from ...measurement import report_stage
from ...model import PreparedComparison, PreparedLaunch, Tolerance
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


def scaled_index_add(context):
    initial = torch.randn((65536, 1, 4096), dtype=torch.float16)
    source = torch.randn((32768, 1, 4096), dtype=torch.float16)
    indices = torch.arange(0, 65536, 2, dtype=torch.int64)
    scaling = torch.randn((4096,), dtype=torch.float16)
    report_stage("generated_compilation")
    artifact = intent.compile(scaled_index_add_unique, target=context.target, compiler=context.compiler,
                              tuning_config=context.tuning_config)
    runtime = load_module(context.project_root / "source/pytorch/cpu_runtime.py", "intent_cpu_reference")

    def side(function):
        output = initial.clone()
        return PreparedLaunch(lambda: function(output, indices, source, scaling, 1.0), lambda: output,
                              prepare=lambda: output.copy_(initial))

    report_stage("adapter_preparation")
    return PreparedComparison(side(artifact.run), side(runtime.scaled_index_add),
        Tolerance(atol=2e-2, rtol=1e-2), cuda_graph=False, device_type="cpu", cpu_host_timing=True,
        note="原65536x1x4096/32768唯一偶数indices f16 scaled-index-add，f32更新后f16写回；独立InOut状态每次恢复且恢复不计时；PyTorch CPU reference，原容差，完整host调用。")


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
         "scaled_index_add": scaled_index_add,
         "scalar_table_lookup": scalar_lookup,
         "shifted_row_copy": shifted_row,
         "roll_rows_forward": rolled_rows,
         "grouped_query_head_add": grouped_query_heads,
         "alternating_signed_indices": alternating_indices}
