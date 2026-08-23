from __future__ import annotations

import torch

from kernels.backward.embedding import embedding_forward_lookup_bf16
from kernels.indexing.relations import index_select_rows
from kernels.indexing.relations import roll_rows_forward
from kernels.indexing.relations import scaled_index_add_unique
from kernels.layout.transpose import matrix_transpose
from ...loading import load_module
from ...measurement import compile_single
from ...measurement import functional_launch
from ...model import Context
from ...model import PreparedComparison
from ...model import PreparedLaunch
from ...model import Tolerance


def _runtime(context: Context, path: str, name: str):
    return load_module(context.project_root / path, name)


def embedding_lookup(context: Context) -> PreparedComparison:
    batch, sequence, vocabulary, hidden = 8, 2048, 32768, 4096
    table = torch.randn(
        (vocabulary, hidden), device="cuda", dtype=torch.bfloat16
    )
    indices = torch.randint(
        0,
        vocabulary,
        (batch, sequence),
        device="cuda",
        dtype=torch.int64,
    )
    _, generated_base = compile_single(
        context,
        embedding_forward_lookup_bf16,
        (table, indices.reshape(-1)),
    )
    generated = PreparedLaunch(
        launch=generated_base.launch,
        outputs=lambda: generated_base.outputs().view(batch, sequence, hidden),
    )
    runtime = _runtime(
        context,
        "source/triton/liger-kernel/embedding/lookup/embedding_runtime.py",
        "intent_v2_triton_embedding_lookup",
    )
    source = functional_launch(
        lambda: runtime.LigerEmbeddingFunction.apply(table, indices)
    )
    return PreparedComparison(generated, source, Tolerance(atol=0.0), cuda_graph=True)


def flaggems_embedding_lookup(context: Context) -> PreparedComparison:
    batch, sequence, vocabulary, hidden = 8, 2048, 32768, 4096
    table = torch.randn(
        (vocabulary, hidden), device="cuda", dtype=torch.bfloat16
    )
    indices = torch.randint(
        0, vocabulary, (batch, sequence), device="cuda", dtype=torch.int64
    )
    _, generated_base = compile_single(
        context,
        embedding_forward_lookup_bf16,
        (table, indices.reshape(-1)),
    )
    generated = PreparedLaunch(
        launch=generated_base.launch,
        outputs=lambda: generated_base.outputs().view(batch, sequence, hidden),
    )
    runtime = _runtime(
        context,
        "source/triton/flag-gems/indexing/embedding/embedding_runtime.py",
        "intent_v2_triton_flaggems_embedding",
    )
    source = functional_launch(lambda: runtime.upstream((indices, table)))
    return PreparedComparison(generated, source, Tolerance(atol=0.0), cuda_graph=True)


def flaggems_roll(context: Context) -> PreparedComparison:
    x = torch.randn((8192, 4096), device="cuda", dtype=torch.float16)
    _, generated = compile_single(context, roll_rows_forward, (x,))
    runtime = _runtime(
        context,
        "source/triton/flag-gems/indexing/roll/roll_runtime.py",
        "intent_v2_triton_flaggems_roll",
    )
    source = functional_launch(lambda: runtime.upstream((x,)))
    return PreparedComparison(generated, source, Tolerance(atol=0.0), cuda_graph=True)


def flaggems_transpose_copy(context: Context) -> PreparedComparison:
    x = torch.randn((4093, 8191), device="cuda", dtype=torch.float16)
    _, generated = compile_single(context, matrix_transpose, (x,))
    runtime = _runtime(
        context,
        "source/triton/flag-gems/layout/copy/copy_runtime.py",
        "intent_v2_triton_flaggems_copy",
    )
    source = functional_launch(lambda: runtime.upstream((x,)))
    return PreparedComparison(generated, source, Tolerance(atol=0.0), cuda_graph=True)


def index_select(context: Context) -> PreparedComparison:
    source_rows, selected_rows, hidden = 65536, 32768, 4096
    source_tensor = torch.randn(
        (source_rows, hidden), device="cuda", dtype=torch.float16
    )
    indices = torch.arange(
        0, selected_rows * 2, 2, device="cuda", dtype=torch.int64
    )
    _, generated = compile_single(
        context,
        index_select_rows,
        (source_tensor, indices),
    )
    runtime = _runtime(
        context,
        "source/triton/xformers/indexing/index_select_cat/k_index_select_cat_runtime.py",
        "intent_v2_triton_index_select",
    )
    source_module = runtime.load_source()
    source_output = torch.empty(
        (selected_rows, hidden), device="cuda", dtype=torch.float16
    )

    def launch():
        source_module.index_select_cat_fwd(source_output, source_tensor, indices)

    launch()
    source = PreparedLaunch(launch=launch, outputs=lambda: source_output)
    return PreparedComparison(generated, source, Tolerance(atol=0.0), cuda_graph=True)


def scaled_index_add(context: Context) -> PreparedComparison:
    destination_rows, routed_rows, hidden = 65536, 32768, 4096
    initial = torch.randn(
        (destination_rows, 1, hidden), device="cuda", dtype=torch.float16
    )
    routed = torch.randn(
        (routed_rows, 1, hidden), device="cuda", dtype=torch.float16
    )
    indices = torch.arange(
        0, routed_rows * 2, 2, device="cuda", dtype=torch.int64
    )
    scaling = torch.randn((hidden,), device="cuda", dtype=torch.float16)
    generated_destination = initial.clone()
    arguments = (generated_destination, indices, routed, scaling, 1.0)
    _, generated_base = compile_single(
        context,
        scaled_index_add_unique,
        arguments,
    )
    generated = PreparedLaunch(
        launch=generated_base.launch,
        outputs=lambda: generated_destination,
        prepare=lambda: generated_destination.copy_(initial),
    )
    runtime = _runtime(
        context,
        "source/triton/xformers/indexing/scaled_index_add/k_scaled_index_add_runtime.py",
        "intent_v2_triton_scaled_index_add",
    )
    source_module = runtime.load_source()
    source_destination = initial.clone()

    def source_launch():
        source_module.scaled_index_add_fwd(
            source_destination,
            indices,
            routed,
            scaling,
            1.0,
        )

    source_launch()
    source = PreparedLaunch(
        launch=source_launch,
        outputs=lambda: source_destination,
        prepare=lambda: source_destination.copy_(initial),
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=2e-2, rtol=1e-2),
        cuda_graph=False,
    )


CASES = {
    "embedding_lookup": embedding_lookup,
    "flaggems_embedding_lookup": flaggems_embedding_lookup,
    "flaggems_roll": flaggems_roll,
    "flaggems_transpose_copy": flaggems_transpose_copy,
    "index_select": index_select,
    "scaled_index_add": scaled_index_add,
}
