"""Data-dependent indices and explicit multi-kernel routing."""

import torch

from kernels.backward.embedding import embedding_forward_lookup_bf16
from kernels.compaction.nonzero import compact_nonzero_rows
from kernels.ragged.jagged_mean import BATCH as JAGGED_BATCH, jagged_mean as jagged_definition
from kernels.routing.moe_align import (
    EXPERTS, ROUTES, PADDED_ROUTES, BLOCK_SIZE,
    moe_count_routes, moe_prefix_routes, moe_scatter_routes, moe_mark_expert_blocks,
)
from kernels.sparse.csr_spmv import (
    ROWS, COLUMNS, NONZEROS, NONZEROS_PER_ROW, csr_spmv as csr_definition,
)
from .composition import MoEAlignment


def embedding(context):
    kernel = context.compile(embedding_forward_lookup_bf16)
    table = torch.randn((1024, 128), device=context.device, dtype=torch.bfloat16)
    indices = torch.randint(0, 1024, (4096,), device=context.device, dtype=torch.int64)
    return context.call(kernel, table, indices)


def csr_spmv(context):
    kernel = context.compile(csr_definition)
    offsets = torch.arange(0, NONZEROS + 1, NONZEROS_PER_ROW,
                           device=context.device, dtype=torch.int32)
    columns = torch.randint(0, COLUMNS, (NONZEROS,), device=context.device, dtype=torch.int32)
    values = torch.randn((NONZEROS,), device=context.device, dtype=torch.float32)
    vector = torch.randn((COLUMNS,), device=context.device, dtype=torch.float32)
    return context.call(kernel, offsets, columns, values, vector)


def jagged_mean(context):
    kernel = context.compile(jagged_definition)
    lengths = torch.arange(JAGGED_BATCH, device=context.device, dtype=torch.int32) % 128 + 1
    offsets = torch.cat((torch.zeros((1,), device=context.device, dtype=torch.int32),
                         lengths.cumsum(0).to(torch.int32)))
    values = torch.randn((33024, 128), device=context.device, dtype=torch.float32)
    return context.call(kernel, values, offsets)


def nonzero(context):
    kernel = context.compile(compact_nonzero_rows)
    values = torch.randn((64, 4096), device=context.device, dtype=torch.float32)
    values[:, ::3] = 0.0
    indices, counts = context.call(kernel, values)
    return {"indices": indices, "counts": counts}


def moe_alignment(context):
    count = context.compile(moe_count_routes)
    prefix = context.compile(moe_prefix_routes)
    scatter = context.compile(moe_scatter_routes)
    mark = context.compile(moe_mark_expert_blocks)
    ids = (torch.arange(ROUTES, device=context.device, dtype=torch.int32)
           .remainder(EXPERTS).reshape(4096, 2))
    program = MoEAlignment(count, prefix, scatter, mark)
    sorted_routes, expert_blocks, total_padded = context.call(program, ids)
    return {"route_ids": sorted_routes, "expert_blocks": expert_blocks,
            "total_padded": total_padded, "block_size": BLOCK_SIZE}
