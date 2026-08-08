from __future__ import annotations

import argparse
import ast
from pathlib import Path

import torch
import tilelang.language as T

import intent

from frontend_softmax import stable_softmax


ROWS = 8192
COLUMNS = 8192


def _load_baseline(source_path: Path):
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    tree.body = [node for node in tree.body if node.end_lineno <= 45]
    namespace = {
        "__file__": str(source_path),
        "__name__": "intent_upstream_tilelang_softmax",
        "M": ROWS,
        "N": COLUMNS,
    }
    exec(compile(tree, str(source_path), "exec"), namespace)
    return namespace["softmax_kernel"].compile(
        BLOCK_M=1,
        BLOCK_N=COLUMNS,
        dtype=T.float32,
    )


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
    artifact = intent.compile(
        stable_softmax,
        target=intent.TileLangTarget(device=0),
        realizer=arguments.intent_realize,
        translator=arguments.intent_translate,
    )
    baseline = _load_baseline(arguments.baseline_source)
    x = torch.randn((ROWS, COLUMNS), device=device, dtype=torch.float32)
    generated = artifact.run(x)
    original = baseline(x)
    reference = torch.softmax(x, dim=1)
    torch.cuda.synchronize()

    generated_error = (generated - reference).abs().max().item()
    original_error = (original - reference).abs().max().item()
    generated_original_error = (generated - original).abs().max().item()
    if generated_error > 1.0e-6 or original_error > 1.0e-6:
        raise RuntimeError(
            "TileLang softmax numerical comparison failed: "
            f"generated/reference={generated_error}, "
            f"upstream/reference={original_error}"
        )

    generated_p50, generated_p95 = _bench(lambda: artifact.run(x))
    original_p50, original_p95 = _bench(lambda: baseline(x))
    print("=== Intent Kernel IR + TileLang Plan MLIR ===")
    print(artifact.mlir, end="")
    print("=== Generated TileLang source ===")
    print(artifact.source, end="")
    print(
        "TileLang softmax numerical comparison: PASS "
        f"(shape=({ROWS}, {COLUMNS}), dtype=f32, "
        f"generated/reference={generated_error}, upstream/reference={original_error}, "
        f"generated/upstream={generated_original_error})"
    )
    print(
        "TileLang softmax wrapper performance: "
        f"upstream_p50={original_p50:.4f} ms, upstream_p95={original_p95:.4f} ms, "
        f"generated_p50={generated_p50:.4f} ms, generated_p95={generated_p95:.4f} ms, "
        f"generated/upstream_p50={generated_p50 / original_p50:.4f}x"
    )


if __name__ == "__main__":
    main()
