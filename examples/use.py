"""Run one of the thirty ordinary host examples; no reference or timing loop."""

import argparse
from importlib import import_module


# This is usage navigation. Names are never sent to the compiler as policy.
EXAMPLES = {
    "relu": ("elementwise", "Pointwise ReLU"),
    "swiglu": ("elementwise", "Two-input SwiGLU"),
    "rope": ("elementwise", "In-place Q/K rotary embedding"),
    "transpose": ("elementwise", "Matrix transpose"),
    "embedding": ("irregular", "BF16 embedding lookup with i64 indices"),
    "csr_spmv": ("irregular", "CSR sparse matrix-vector product"),
    "jagged_mean": ("irregular", "Segmented mean with explicit offsets"),
    "softmax": ("collectives", "FP16 softmax"),
    "layer_norm": ("collectives", "Weighted LayerNorm using the original second moment"),
    "batch_norm": ("collectives", "Welford statistics and mutable running state"),
    "group_norm_backward": ("collectives", "Two-kernel GroupNorm backward"),
    "cumsum": ("collectives", "Ordered row prefix"),
    "causal_linear_attention": ("collectives", "Region scan with matrix state"),
    "nonzero": ("irregular", "Prefix-based compaction and counts"),
    "moe_alignment": ("irregular", "Explicit count/prefix/scatter/mark"),
    "gemm": ("matrix", "FP16 GEMM with an algorithmic activation specialization"),
    "batched_gemm": ("matrix", "Batched BF16 GEMM"),
    "ragged_gemm": ("matrix", "Unequal groups and matrix supply"),
    "int8_gemm": ("matrix", "INT8 GEMM with i32 bias and accumulation"),
    "q4_projection": ("matrix", "Packed Q4_K weights and Q8_K activation quantization"),
    "fp8_gemm": ("matrix", "Block-scaled E4M3 with E8M0 scale storage"),
    "attention": ("streaming", "Causal BF16 grouped-query attention"),
    "paged_decode": ("streaming", "Explicit paged partials and FP32-to-FP16 merge"),
    "mamba_chunk_state": ("streaming", "BF16 matrix supply and FP32 chunk state"),
    "gated_delta": ("streaming", "Ordered gated recurrence and final state"),
    "causal_convolution": ("structured", "Causal depthwise convolution"),
    "cholesky": ("structured", "In-place small batched Cholesky"),
    "dropout": ("elementwise", "Author's integer-mixing dropout"),
    "adamw": ("elementwise", "Mutable parameters and optimizer moments"),
    "histogram": ("collectives", "Histogram with out-of-range sample masking"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", nargs="?", choices=tuple(EXAMPLES))
    parser.add_argument("--list", action="store_true", help="List usage examples without importing a provider")
    parser.add_argument("--target", choices=("triton", "cutile", "mojo"), default="triton")
    parser.add_argument("--device", type=int, default=0, help="CUDA device for a GPU target")
    parser.add_argument("--compiler", help="Use an explicitly selected intent-compile")
    parser.add_argument("--prepared", action="store_true", help="Bind each kernel invocation before launch")
    args = parser.parse_args()
    if args.list:
        for name, (_, description) in EXAMPLES.items():
            print(f"{name:24} {description}")
        return
    if args.example is None:
        parser.error("select an example or use --list")

    import intent
    import torch
    from programs.common import ExampleContext, describe_outputs

    if args.target == "mojo":
        target = intent.MojoTarget()
        device = torch.device("cpu")
    else:
        target_class = intent.TritonTarget if args.target == "triton" else intent.CuTileTarget
        target = target_class(device=args.device)
        device = torch.device("cuda", args.device)
    context = ExampleContext(target, device, compiler=args.compiler, prepared=args.prepared)
    module, _ = EXAMPLES[args.example]
    function = getattr(import_module(f"programs.{module}"), args.example)
    result = function(context)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    describe_outputs(result)
    for artifact in context.artifacts:
        print(f"Generated source, IR and diagnostics: {artifact.cache_directory}")


if __name__ == "__main__":
    main()
