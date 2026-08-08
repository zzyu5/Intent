from __future__ import annotations

import argparse
import ast
import importlib.util
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import tilelang.language as T

import intent
from kernels.attention import BATCH
from kernels.attention import HEAD_DIMENSION
from kernels.attention import HEADS
from kernels.attention import SCALE
from kernels.attention import SEQUENCE
from kernels.attention import flash_attention_fwd
from kernels.gemm import Activation
from kernels.gemm import K
from kernels.gemm import M
from kernels.gemm import N
from kernels.gemm import gemm
from kernels.softmax import COLUMNS
from kernels.softmax import ROWS
from kernels.softmax import stable_softmax
from repro.common.support import benchmark
from repro.common.support import print_artifact


def _load_module(source_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_softmax(compiler: str, baseline_source: Path) -> None:
    tree = ast.parse(baseline_source.read_text(), filename=str(baseline_source))
    tree.body = [node for node in tree.body if node.end_lineno <= 45]
    namespace = {
        "__file__": str(baseline_source),
        "__name__": "intent_upstream_tilelang_softmax",
        "M": ROWS,
        "N": COLUMNS,
    }
    exec(compile(tree, str(baseline_source), "exec"), namespace)
    baseline = namespace["softmax_kernel"].compile(
        BLOCK_M=1,
        BLOCK_N=COLUMNS,
        dtype=T.float32,
    )
    artifact = intent.compile(
        stable_softmax,
        target=intent.TileLangTarget(device=0),
        compiler=compiler,
    )
    x = torch.randn((ROWS, COLUMNS), device="cuda", dtype=torch.float32)
    generated = artifact.run(x)
    upstream = baseline(x)
    reference = torch.softmax(x, dim=1)
    torch.cuda.synchronize()
    generated_error = (generated - reference).abs().max().item()
    upstream_error = (upstream - reference).abs().max().item()
    generated_upstream_error = (generated - upstream).abs().max().item()
    if generated_error > 1.0e-6 or upstream_error > 1.0e-6:
        raise RuntimeError(
            "TileLang softmax numerical comparison failed: "
            f"generated/reference={generated_error}, "
            f"upstream/reference={upstream_error}"
        )
    generated_p50, generated_p95 = benchmark(lambda: artifact.run(x))
    upstream_p50, upstream_p95 = benchmark(lambda: baseline(x))
    print_artifact(artifact, "TileLang")
    print(
        "TileLang softmax numerical comparison: PASS "
        f"(shape=({ROWS}, {COLUMNS}), dtype=f32, "
        f"generated/reference={generated_error}, upstream/reference={upstream_error}, "
        f"generated/upstream={generated_upstream_error})"
    )
    print(
        "TileLang softmax wrapper performance: "
        f"upstream_p50={upstream_p50:.4f} ms, upstream_p95={upstream_p95:.4f} ms, "
        f"generated_p50={generated_p50:.4f} ms, generated_p95={generated_p95:.4f} ms, "
        f"generated/upstream_p50={generated_p50 / upstream_p50:.4f}x"
    )


def _run_gemm(compiler: str, baseline_source: Path) -> None:
    baseline = _load_module(
        baseline_source, "intent_upstream_tilelang_gemm"
    ).matmul.compile(
        M=M,
        N=N,
        K=K,
        block_M=128,
        block_N=128,
        block_K=32,
    )
    artifact = intent.compile(
        gemm,
        constexprs={"ACTIVATION": Activation.NONE},
        target=intent.TileLangTarget(device=0),
        compiler=compiler,
    )
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    a /= math.sqrt(K)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    generated = artifact.run(a, b)
    upstream = baseline(a, b)
    reference = torch.matmul(a, b)
    torch.cuda.synchronize()
    generated_error = (generated - reference).abs().max().item()
    upstream_error = (upstream - reference).abs().max().item()
    generated_upstream_error = (generated - upstream).abs().max().item()
    if generated_error > 2.0e-2 or upstream_error > 2.0e-2:
        raise RuntimeError(
            "TileLang GEMM numerical comparison failed: "
            f"generated/reference={generated_error}, "
            f"upstream/reference={upstream_error}"
        )
    generated_p50, generated_p95 = benchmark(lambda: artifact.run(a, b))
    upstream_p50, upstream_p95 = benchmark(lambda: baseline(a, b))
    print_artifact(artifact, "TileLang")
    print(
        "TileLang GEMM numerical comparison: PASS "
        f"(shape=({M}, {K}) x ({K}, {N}), dtype=f16, "
        f"generated/reference={generated_error}, upstream/reference={upstream_error}, "
        f"generated/upstream={generated_upstream_error})"
    )
    print(
        "TileLang GEMM wrapper performance: "
        f"upstream_p50={upstream_p50:.4f} ms, upstream_p95={upstream_p95:.4f} ms, "
        f"generated_p50={generated_p50:.4f} ms, generated_p95={generated_p95:.4f} ms, "
        f"generated/upstream_p50={generated_p50 / upstream_p50:.4f}x"
    )


def _run_attention(compiler: str, baseline_source: Path) -> None:
    baseline = _load_module(
        baseline_source, "intent_upstream_tilelang_attention"
    ).flashattn(
        BATCH,
        HEADS,
        SEQUENCE,
        HEAD_DIMENSION,
        False,
        block_M=128,
        block_N=128,
        num_stages=1,
        threads=128,
    )
    artifact = intent.compile(
        flash_attention_fwd,
        constexprs={"CAUSAL": False},
        target=intent.TileLangTarget(device=0),
        compiler=compiler,
    )
    shape = (BATCH, HEADS, SEQUENCE, HEAD_DIMENSION)
    q = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.5
    k = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.5
    v = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.5
    q_bshd = q.permute(0, 2, 1, 3).contiguous()
    k_bshd = k.permute(0, 2, 1, 3).contiguous()
    v_bshd = v.permute(0, 2, 1, 3).contiguous()
    generated = artifact.run(q, k, v, SCALE)
    upstream_bshd = baseline(q_bshd, k_bshd, v_bshd)
    upstream = upstream_bshd.permute(0, 2, 1, 3)
    reference = F.scaled_dot_product_attention(
        q, k, v, is_causal=False, scale=SCALE
    )
    torch.cuda.synchronize()
    generated_error = (generated - reference).abs().max().item()
    upstream_error = (upstream - reference).abs().max().item()
    generated_upstream_error = (generated - upstream).abs().max().item()
    if generated_error > 2.0e-2 or upstream_error > 2.0e-2:
        raise RuntimeError(
            "TileLang attention numerical comparison failed: "
            f"generated/reference={generated_error}, "
            f"upstream/reference={upstream_error}"
        )
    generated_p50, generated_p95 = benchmark(
        lambda: artifact.run(q, k, v, SCALE)
    )
    upstream_p50, upstream_p95 = benchmark(
        lambda: baseline(q_bshd, k_bshd, v_bshd)
    )
    print_artifact(artifact, "TileLang")
    print(
        "TileLang attention numerical comparison: PASS "
        f"(shape={shape}, dtype=f16, causal=False, "
        f"generated/reference={generated_error}, upstream/reference={upstream_error}, "
        f"generated/upstream={generated_upstream_error})"
    )
    print(
        "TileLang attention wrapper performance: "
        f"upstream_p50={upstream_p50:.4f} ms, upstream_p95={upstream_p95:.4f} ms, "
        f"generated_p50={generated_p50:.4f} ms, generated_p95={generated_p95:.4f} ms, "
        f"generated/upstream_p50={generated_p50 / upstream_p50:.4f}x"
    )


RUNNERS = {
    "attention": _run_attention,
    "gemm": _run_gemm,
    "softmax": _run_softmax,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", choices=sorted(RUNNERS))
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    arguments = parser.parse_args()
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    RUNNERS[arguments.kernel](arguments.compiler, arguments.baseline_source)


if __name__ == "__main__":
    main()
