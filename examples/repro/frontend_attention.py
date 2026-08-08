from __future__ import annotations

import argparse
import ast
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

import intent

from frontend_softmax import flash_attention_fwd


BATCH = 4
HEADS = 32
SEQUENCE = 4096
HEAD_DIMENSION = 128
SCALE = 0.5


def _load_baseline(source_path: Path):
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    tree.body = [node for node in tree.body if node.end_lineno <= 621]
    namespace = {
        "__file__": str(source_path),
        "__name__": "intent_original_triton_attention",
    }
    exec(compile(tree, str(source_path), "exec"), namespace)
    return namespace["attention"]


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
        flash_attention_fwd,
        constexprs={"CAUSAL": False},
        target=intent.TritonTarget(device=0),
        realizer=arguments.intent_realize,
        translator=arguments.intent_translate,
    )
    baseline = _load_baseline(arguments.baseline_source)
    shape = (BATCH, HEADS, SEQUENCE, HEAD_DIMENSION)
    q = torch.randn(shape, device=device, dtype=torch.float16) * 0.5
    k = torch.randn(shape, device=device, dtype=torch.float16) * 0.5
    v = torch.randn(shape, device=device, dtype=torch.float16) * 0.5
    generated = torch.empty_like(q)
    artifact(q, k, v, generated, SCALE)
    original = baseline(q, k, v, False, SCALE, False)
    reference = F.scaled_dot_product_attention(
        q, k, v, is_causal=False, scale=SCALE
    )
    stream.synchronize()

    generated_error = torch.max(torch.abs(generated - reference)).item()
    original_error = torch.max(torch.abs(original - reference)).item()
    generated_original_error = torch.max(torch.abs(generated - original)).item()
    if generated_error > 2.0e-2 or generated_original_error > 2.0e-2:
        raise RuntimeError(
            "attention numerical comparison failed: "
            f"generated/reference={generated_error}, "
            f"generated/original={generated_original_error}"
        )

    original_p50, original_p95 = triton.testing.do_bench(
        lambda: baseline(q, k, v, False, SCALE, False),
        warmup=25,
        rep=100,
        quantiles=[0.5, 0.95],
    )
    generated_p50, generated_p95 = triton.testing.do_bench(
        lambda: artifact.run(q, k, v, SCALE),
        warmup=25,
        rep=100,
        quantiles=[0.5, 0.95],
    )

    print("=== Intent Kernel IR + Triton Plan MLIR ===")
    print(artifact.mlir, end="")
    print("=== Generated Triton source ===")
    print(artifact.source, end="")
    print("backend IR levels: " + ", ".join(sorted(artifact.backend_ir)))
    print(
        "attention numerical comparison: PASS "
        f"(shape={shape}, dtype=f16, causal=False, "
        f"generated/reference={generated_error}, original/reference={original_error}, "
        f"generated/original={generated_original_error})"
    )
    print(
        "attention wrapper performance (includes output allocation): "
        f"original_p50={original_p50:.4f} ms, original_p95={original_p95:.4f} ms, "
        f"generated_p50={generated_p50:.4f} ms, generated_p95={generated_p95:.4f} ms, "
        f"generated/original_p50={generated_p50 / original_p50:.4f}x"
    )


if __name__ == "__main__":
    main()
