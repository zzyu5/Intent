from __future__ import annotations

from collections.abc import Callable
import inspect
from types import FunctionType

import torch

from intent.runtime import CompiledArtifact
from kernels.ragged.moe import EXPERTS
from kernels.ragged.moe import HIDDEN
from kernels.ragged.moe import TOKENS
from kernels.ragged.moe import TOP_K


def benchmark(
    function: Callable[[], object],
    *,
    warmup: int = 25,
    repetitions: int = 100,
    cuda_graph: bool = False,
    prepare: Callable[[], object] | None = None,
) -> tuple[float, float]:
    if cuda_graph and prepare is not None:
        raise ValueError("CUDA Graph measurement cannot reset inputs between replays")
    measured = function
    measurement_stream = torch.cuda.current_stream()
    if cuda_graph:
        measurement_stream = torch.cuda.Stream()
        measurement_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(measurement_stream):
            for _ in range(warmup):
                function()
        measurement_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=measurement_stream):
            function()
        measured = graph.replay
    else:
        for _ in range(warmup):
            if prepare is not None:
                prepare()
            function()
    flush_buffer = None
    if cuda_graph:
        l2_bytes = torch.cuda.get_device_properties(
            measurement_stream.device
        ).L2_cache_size
        if l2_bytes <= 0:
            raise RuntimeError("CUDA device does not report a positive L2 cache size")
        flush_buffer = torch.empty(
            (2 * l2_bytes) // 4,
            device=measurement_stream.device,
            dtype=torch.int32,
        )
        with torch.cuda.stream(measurement_stream):
            flush_buffer.add_(1)
        measurement_stream.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    with torch.cuda.stream(measurement_stream):
        for start, end in zip(starts, ends):
            if flush_buffer is not None:
                flush_buffer.add_(1)
            if prepare is not None:
                prepare()
            start.record()
            measured()
            end.record()
    measurement_stream.synchronize()
    samples = torch.tensor(
        [start.elapsed_time(end) for start, end in zip(starts, ends)]
    )
    return samples.quantile(0.5).item(), samples.quantile(0.95).item()


def prepare_kernel_call(
    artifact: CompiledArtifact,
    arguments: tuple[object, ...],
    outputs: object,
) -> Callable[[], object]:
    run_arguments = inspect.signature(artifact._runner).bind(*arguments).arguments
    launch_parameters = inspect.signature(artifact._launcher).parameters
    output_values = outputs if isinstance(outputs, tuple) else (outputs,)
    output_names = [name for name in launch_parameters if name not in run_arguments]
    if len(output_names) != len(output_values):
        raise RuntimeError(
            "generated launch ABI does not match the prepared output buffers"
        )
    launch_arguments = dict(run_arguments)
    launch_arguments.update(zip(output_names, output_values))
    ordered = tuple(launch_arguments[name] for name in launch_parameters)
    compiled = artifact._launcher(*ordered)
    if isinstance(compiled, FunctionType):
        return compiled
    if callable(compiled):
        return lambda: compiled(*ordered)
    return lambda: artifact._launcher(*ordered)


def print_artifact(artifact: CompiledArtifact, target: str) -> None:
    print(f"=== Intent Kernel IR + {target} Plan MLIR ===")
    print(artifact.mlir, end="")
    print(f"=== Generated {target} source ===")
    print(artifact.source, end="")
    if artifact.backend_ir:
        print("backend IR levels: " + ", ".join(sorted(artifact.backend_ir)))


def make_moe_routes(device: torch.device):
    route_count = TOKENS * TOP_K
    route_token = torch.arange(
        TOKENS, device=device, dtype=torch.int32
    ).repeat_interleave(TOP_K)
    route_weights = torch.full(
        (route_count,), 1.0 / TOP_K, device=device, dtype=torch.float32
    )
    topk_ids = (torch.arange(route_count, device=device) % EXPERTS).reshape(
        TOKENS, TOP_K
    )
    flat_experts = topk_ids.reshape(-1)
    member_routes = torch.argsort(flat_experts, stable=True).to(torch.int32)
    counts = torch.bincount(flat_experts, minlength=EXPERTS)
    route_offsets = torch.cat(
        (torch.zeros(1, device=device, dtype=torch.int64), counts.cumsum(0))
    ).to(torch.int32)
    return route_offsets, member_routes, route_token, route_weights, topk_ids


def moe_reference(
    x,
    route_offsets,
    member_routes,
    route_token,
    route_weights,
    w1,
    w2,
):
    result = torch.zeros((TOKENS, HIDDEN), device=x.device, dtype=torch.float32)
    for expert in range(EXPERTS):
        routes = member_routes[
            route_offsets[expert] : route_offsets[expert + 1]
        ].long()
        tokens = route_token[routes].long()
        hidden = torch.relu(x[tokens].float() @ w1[expert].float())
        values = hidden @ w2[expert].float()
        result.index_add_(0, tokens, route_weights[routes, None] * values)
    return result
