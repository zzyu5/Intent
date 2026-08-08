from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path

import torch
import triton

import intent
import intent.language as I


GEMM_M = 4096
GEMM_K = 4096
GEMM_N = 14336


class Activation(I.Enum):
    NONE = 0
    RELU = 1


@intent.kernel
def gemm(
    a: I.In[I.f16, ("M", "K")],
    b: I.In[I.f16, ("K", "N")],
    c: I.Out[I.f16, ("M", "N")],
    ACTIVATION: I.Constexpr[Activation],
):
    M, K = a.shape
    _, N = b.shape
    m_axis = I.domain(0, M)
    n_axis = I.domain(0, N)
    k_axis = I.domain(0, K)
    for mr in I.parallel(I.partition(m_axis, extent=I.auto("M_TILE"))):
        for nr in I.parallel(I.partition(n_axis, extent=I.auto("N_TILE"))):
            accumulator = I.contract(
                a[mr, k_axis],
                b[k_axis, nr],
                reduce=((1, 0),),
                acc_dtype=I.f32,
            )
            if ACTIVATION == Activation.RELU:
                accumulator = I.maximum(accumulator, 0.0)
            c[mr, nr] = I.cast(accumulator, I.f16)


def _load_original_gemm(source_path: Path):
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    tree.body = [node for node in tree.body if node.end_lineno <= 352]
    namespace = {
        "__file__": str(source_path),
        "__name__": "intent_original_triton_gemm",
    }
    exec(compile(tree, str(source_path), "exec"), namespace)
    return namespace["matmul"]


def _run_gemm(
    intent_realize: str,
    intent_translate: str,
    baseline_source: Path,
) -> None:
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    comparison_stream = torch.cuda.Stream(device=device)
    torch.cuda.set_stream(comparison_stream)
    artifact = intent.compile(
        gemm,
        constexprs={"ACTIVATION": Activation.NONE},
        target=intent.TritonTarget(device=0),
        realizer=intent_realize,
        translator=intent_translate,
    )
    original_gemm = _load_original_gemm(baseline_source)
    a = torch.randn((GEMM_M, GEMM_K), device=device, dtype=torch.float16)
    a /= math.sqrt(GEMM_K)
    b = torch.randn((GEMM_K, GEMM_N), device=device, dtype=torch.float16)
    generated_output = torch.empty(
        (GEMM_M, GEMM_N), device=device, dtype=torch.float16
    )
    artifact(a, b, generated_output)
    original_output = original_gemm(a, b)
    reference = torch.matmul(a, b)
    comparison_stream.synchronize()

    generated_error = torch.max(torch.abs(generated_output - reference)).item()
    original_error = torch.max(torch.abs(original_output - reference)).item()
    generated_original_error = torch.max(
        torch.abs(generated_output - original_output)
    ).item()
    if generated_error > 2.0e-2 or generated_original_error > 2.0e-2:
        raise RuntimeError(
            "GEMM numerical comparison failed: "
            f"generated/reference={generated_error}, "
            f"generated/original={generated_original_error}"
        )

    original_p50, original_p95 = triton.testing.do_bench(
        lambda: original_gemm(a, b),
        warmup=25,
        rep=100,
        quantiles=[0.5, 0.95],
    )
    generated_p50, generated_p95 = triton.testing.do_bench(
        lambda: artifact.run(a, b),
        warmup=25,
        rep=100,
        quantiles=[0.5, 0.95],
    )
    if generated_p50 > original_p50 * 1.05:
        raise RuntimeError(
            "GEMM performance regressed by more than 5%: "
            f"generated={generated_p50:.4f} ms, original={original_p50:.4f} ms"
        )

    print("=== Intent Kernel IR + Physical Plan MLIR ===")
    print(artifact.mlir, end="")
    print("=== Generated Triton source ===")
    print(artifact.source, end="")
    print("backend IR levels: " + ", ".join(sorted(artifact.backend_ir)))
    print(
        "GEMM numerical comparison: PASS "
        f"(shape=({GEMM_M}, {GEMM_K}) x ({GEMM_K}, {GEMM_N}), dtype=f16, "
        f"generated/reference={generated_error}, original/reference={original_error}, "
        f"generated/original={generated_original_error})"
    )
    print(
        "GEMM wrapper performance (includes output allocation, cuda:0, one stream): "
        f"original_p50={original_p50:.4f} ms, original_p95={original_p95:.4f} ms, "
        f"generated_p50={generated_p50:.4f} ms, generated_p95={generated_p95:.4f} ms, "
        f"generated/original_p50={generated_p50 / original_p50:.4f}x, "
        f"generated/original_p95={generated_p95 / original_p95:.4f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--intent-realize",
        default="/tmp/intentdsl-build/tools/intent-realize/intent-realize",
    )
    parser.add_argument(
        "--intent-translate",
        default="/tmp/intentdsl-build/tools/intent-translate/intent-translate",
    )
    parser.add_argument("--baseline-source", type=Path, required=True)
    arguments = parser.parse_args()
    _run_gemm(
        arguments.intent_realize,
        arguments.intent_translate,
        arguments.baseline_source,
    )


if __name__ == "__main__":
    main()
