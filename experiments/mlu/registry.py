from dataclasses import replace

from experiments.gpu.registry import TRITON, CUTILE
from experiments.cpu.registry import MOJO


BANGC = tuple(entry for entry in CUTILE if entry.kernel in (
    "relu", "chunk_gated_delta", "recurrent_gated_delta")) + tuple(
    entry for entry in TRITON if entry.kernel in (
        "fused_softmax", "dense_gemm", "jagged_mean", "flaggems_triangular_solve",
        "mamba_chunk_state", "mamba_state_passing", "mamba_chunk_scan", "paged_gqa_decode")
) + tuple(entry for entry in MOJO if entry.kernel in (
    "causal_attention_f32", "causal_linear_attention_f32"))

BANGC = tuple(replace(entry, source_runtime="experiments/mlu/baselines/cnnl/runtime.py")
              if entry.kernel in ("relu", "fused_softmax") else entry for entry in BANGC)


BY_PROVIDER = {
    "bangc": BANGC,
}
