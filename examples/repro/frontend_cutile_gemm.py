from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch

import intent

from frontend_gemm import Activation
from frontend_gemm import GEMM_K
from frontend_gemm import GEMM_M
from frontend_gemm import GEMM_N
from frontend_gemm import gemm


def _load_baseline(source_path: Path):
    spec = importlib.util.spec_from_file_location(
        "intent_original_cutile_gemm", source_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.matmul


def _bench(function, *, warmup: int = 25, repetitions: int = 100):
    for _ in range(warmup):
        function()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    for start, end in zip(starts, ends):
        start.record()
        function()
        end.record()
    torch.cuda.synchronize()
    samples = torch.tensor([start.elapsed_time(end) for start, end in zip(starts, ends)])
    return samples.quantile(0.5).item(), samples.quantile(0.95).item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intent-realize", required=True)
    parser.add_argument("--intent-translate", required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    arguments = parser.parse_args()

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    stream = torch.cuda.Stream(device=device)
    torch.cuda.set_stream(stream)
    artifact = intent.compile(
        gemm,
        constexprs={"ACTIVATION": Activation.NONE},
        target=intent.CuTileTarget(device=0),
        realizer=arguments.intent_realize,
        translator=arguments.intent_translate,
    )
    baseline = _load_baseline(arguments.baseline_source)
    a = torch.randn((GEMM_M, GEMM_K), device=device, dtype=torch.float16)
    a /= math.sqrt(GEMM_K)
    b = torch.randn((GEMM_K, GEMM_N), device=device, dtype=torch.float16)
    generated = torch.empty((GEMM_M, GEMM_N), device=device, dtype=torch.float16)
    artifact(a, b, generated)
    original = baseline(a, b)
    reference = torch.matmul(a, b)
    stream.synchronize()

    generated_error = torch.max(torch.abs(generated - reference)).item()
    original_error = torch.max(torch.abs(original - reference)).item()
    generated_original_error = torch.max(torch.abs(generated - original)).item()
    if generated_error > 2.0e-2 or generated_original_error > 2.0e-2:
        raise RuntimeError(
            "cuTile GEMM numerical comparison failed: "
            f"generated/reference={generated_error}, "
            f"generated/original={generated_original_error}"
        )

    original_p50, original_p95 = _bench(lambda: baseline(a, b))
    generated_p50, generated_p95 = _bench(lambda: artifact.run(a, b))
    if generated_p50 > original_p50 * 1.05:
        raise RuntimeError(
            "cuTile GEMM performance regressed by more than 5%: "
            f"generated={generated_p50:.4f} ms, original={original_p50:.4f} ms"
        )

    print("=== Intent Kernel IR + cuTile Plan MLIR ===")
    print(artifact.mlir, end="")
    print("=== Generated cuTile source ===")
    print(artifact.source, end="")
    print(
        "cuTile GEMM numerical comparison: PASS "
        f"(shape=({GEMM_M}, {GEMM_K}) x ({GEMM_K}, {GEMM_N}), dtype=f16, "
        f"generated/reference={generated_error}, original/reference={original_error}, "
        f"generated/original={generated_original_error})"
    )
    print(
        "cuTile GEMM wrapper performance (includes output allocation): "
        f"original_p50={original_p50:.4f} ms, original_p95={original_p95:.4f} ms, "
        f"generated_p50={generated_p50:.4f} ms, generated_p95={generated_p95:.4f} ms, "
        f"generated/original_p50={generated_p50 / original_p50:.4f}x"
    )


if __name__ == "__main__":
    main()
