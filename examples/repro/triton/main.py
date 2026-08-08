from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path

import torch
import torch.nn.functional as F

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
from kernels.moe import EXPERTS
from kernels.moe import HIDDEN
from kernels.moe import INTERMEDIATE
from kernels.moe import TOKENS
from kernels.moe import TOP_K
from kernels.moe import moe_expert_ffn
from kernels.softmax import COLUMNS
from kernels.softmax import ROWS
from kernels.softmax import stable_softmax
from repro.common.support import benchmark
from repro.common.support import make_moe_routes
from repro.common.support import moe_reference
from repro.common.support import print_artifact


def _load_prefix(source_path: Path, last_line: int, symbol: str, module_name: str):
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    tree.body = [node for node in tree.body if node.end_lineno <= last_line]
    namespace = {"__file__": str(source_path), "__name__": module_name}
    exec(compile(tree, str(source_path), "exec"), namespace)
    return namespace[symbol]


def _run_softmax(compiler: str, baseline_source: Path) -> None:
    artifact = intent.compile(
        stable_softmax,
        target=intent.TritonTarget(device=0),
        compiler=compiler,
    )
    baseline = _load_prefix(
        baseline_source, 175, "softmax", "intent_upstream_triton_softmax"
    )
    x = torch.randn((ROWS, COLUMNS), device="cuda", dtype=torch.float32)
    generated = artifact.run(x)
    upstream = baseline(x)
    reference = torch.softmax(x, dim=1)
    torch.cuda.synchronize()
    generated_error = (generated - reference).abs().max().item()
    upstream_error = (upstream - reference).abs().max().item()
    generated_upstream_error = (generated - upstream).abs().max().item()
    if generated_error > 1.0e-5 or generated_upstream_error > 1.0e-6:
        raise RuntimeError(
            "Triton softmax numerical comparison failed: "
            f"generated/reference={generated_error}, "
            f"generated/upstream={generated_upstream_error}"
        )
    upstream_p50, upstream_p95 = benchmark(lambda: baseline(x))
    generated_p50, generated_p95 = benchmark(lambda: artifact.run(x))
    print_artifact(artifact, "Triton")
    print(
        "Triton softmax numerical comparison: PASS "
        f"(shape=({ROWS}, {COLUMNS}), dtype=f32, "
        f"generated/reference={generated_error}, upstream/reference={upstream_error}, "
        f"generated/upstream={generated_upstream_error})"
    )
    print(
        "Triton softmax wrapper performance: "
        f"upstream_p50={upstream_p50:.4f} ms, upstream_p95={upstream_p95:.4f} ms, "
        f"generated_p50={generated_p50:.4f} ms, generated_p95={generated_p95:.4f} ms, "
        f"generated/upstream_p50={generated_p50 / upstream_p50:.4f}x"
    )


def _run_gemm(compiler: str, baseline_source: Path) -> None:
    artifact = intent.compile(
        gemm,
        constexprs={"ACTIVATION": Activation.NONE},
        target=intent.TritonTarget(device=0),
        compiler=compiler,
    )
    baseline = _load_prefix(
        baseline_source, 352, "matmul", "intent_upstream_triton_gemm"
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
    if generated_error > 2.0e-2 or generated_upstream_error > 2.0e-2:
        raise RuntimeError(
            "Triton GEMM numerical comparison failed: "
            f"generated/reference={generated_error}, "
            f"generated/upstream={generated_upstream_error}"
        )
    upstream_p50, upstream_p95 = benchmark(lambda: baseline(a, b))
    generated_p50, generated_p95 = benchmark(lambda: artifact.run(a, b))
    if generated_p50 > upstream_p50 * 1.05:
        raise RuntimeError(
            "Triton GEMM performance regressed by more than 5%: "
            f"generated={generated_p50:.4f} ms, upstream={upstream_p50:.4f} ms"
        )
    print_artifact(artifact, "Triton")
    print(
        "Triton GEMM numerical comparison: PASS "
        f"(shape=({M}, {K}) x ({K}, {N}), dtype=f16, "
        f"generated/reference={generated_error}, upstream/reference={upstream_error}, "
        f"generated/upstream={generated_upstream_error})"
    )
    print(
        "Triton GEMM wrapper performance: "
        f"upstream_p50={upstream_p50:.4f} ms, upstream_p95={upstream_p95:.4f} ms, "
        f"generated_p50={generated_p50:.4f} ms, generated_p95={generated_p95:.4f} ms, "
        f"generated/upstream_p50={generated_p50 / upstream_p50:.4f}x"
    )


def _run_attention(compiler: str, baseline_source: Path) -> None:
    artifact = intent.compile(
        flash_attention_fwd,
        constexprs={"CAUSAL": False},
        target=intent.TritonTarget(device=0),
        compiler=compiler,
    )
    baseline = _load_prefix(
        baseline_source, 621, "attention", "intent_upstream_triton_attention"
    )
    shape = (BATCH, HEADS, SEQUENCE, HEAD_DIMENSION)
    q = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.5
    k = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.5
    v = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.5
    generated = artifact.run(q, k, v, SCALE)
    upstream = baseline(q, k, v, False, SCALE, False)
    reference = F.scaled_dot_product_attention(
        q, k, v, is_causal=False, scale=SCALE
    )
    torch.cuda.synchronize()
    generated_error = (generated - reference).abs().max().item()
    upstream_error = (upstream - reference).abs().max().item()
    generated_upstream_error = (generated - upstream).abs().max().item()
    if generated_error > 2.0e-2 or generated_upstream_error > 2.0e-2:
        raise RuntimeError(
            "Triton attention numerical comparison failed: "
            f"generated/reference={generated_error}, "
            f"generated/upstream={generated_upstream_error}"
        )
    upstream_p50, upstream_p95 = benchmark(
        lambda: baseline(q, k, v, False, SCALE, False)
    )
    generated_p50, generated_p95 = benchmark(
        lambda: artifact.run(q, k, v, SCALE)
    )
    print_artifact(artifact, "Triton")
    print(
        "Triton attention numerical comparison: PASS "
        f"(shape={shape}, dtype=f16, causal=False, "
        f"generated/reference={generated_error}, upstream/reference={upstream_error}, "
        f"generated/upstream={generated_upstream_error})"
    )
    print(
        "Triton attention wrapper performance: "
        f"upstream_p50={upstream_p50:.4f} ms, upstream_p95={upstream_p95:.4f} ms, "
        f"generated_p50={generated_p50:.4f} ms, generated_p95={generated_p95:.4f} ms, "
        f"generated/upstream_p50={generated_p50 / upstream_p50:.4f}x"
    )


def _triton_moe_baseline(
    grouped_gemm,
    x,
    route_offsets,
    member_routes,
    route_token,
    route_weights,
    w1,
    w2,
):
    routes_by_expert = [
        member_routes[route_offsets[e] : route_offsets[e + 1]].long()
        for e in range(EXPERTS)
    ]
    tokens_by_expert = [route_token[routes].long() for routes in routes_by_expert]
    hidden = grouped_gemm(
        [x[tokens] for tokens in tokens_by_expert],
        [w1[e] for e in range(EXPERTS)],
    )
    output = grouped_gemm(
        [torch.relu(value) for value in hidden],
        [w2[e] for e in range(EXPERTS)],
    )
    merged = torch.zeros((TOKENS, HIDDEN), device=x.device, dtype=torch.float32)
    for routes, tokens, values in zip(routes_by_expert, tokens_by_expert, output):
        merged.index_add_(0, tokens, route_weights[routes, None] * values.float())
    return merged


def _run_moe(compiler: str, baseline_source: Path) -> None:
    artifact = intent.compile(
        moe_expert_ffn,
        target=intent.TritonTarget(device=0),
        compiler=compiler,
    )
    baseline = _load_prefix(
        baseline_source, 213, "group_gemm_fn", "intent_upstream_triton_moe"
    )
    device = torch.device("cuda", 0)
    route_offsets, member_routes, route_token, route_weights, _ = make_moe_routes(
        device
    )
    x = torch.randn((TOKENS, HIDDEN), device=device, dtype=torch.float16)
    x /= math.sqrt(HIDDEN)
    w1 = torch.randn(
        (EXPERTS, HIDDEN, INTERMEDIATE), device=device, dtype=torch.float16
    )
    w1 /= math.sqrt(HIDDEN)
    w2 = torch.randn(
        (EXPERTS, INTERMEDIATE, HIDDEN), device=device, dtype=torch.float16
    )
    w2 /= math.sqrt(INTERMEDIATE)
    generated = artifact.run(
        x, route_offsets, member_routes, route_token, route_weights, w1, w2
    )
    upstream = _triton_moe_baseline(
        baseline,
        x,
        route_offsets,
        member_routes,
        route_token,
        route_weights,
        w1,
        w2,
    )
    reference = moe_reference(
        x, route_offsets, member_routes, route_token, route_weights, w1, w2
    )
    torch.cuda.synchronize()
    generated_error = (generated - reference).abs().max().item()
    upstream_error = (upstream - reference).abs().max().item()
    if generated_error > 2.0e-3:
        raise RuntimeError(
            f"Triton MoE numerical comparison failed: {generated_error}"
        )
    generated_p50, generated_p95 = benchmark(
        lambda: artifact.run(
            x, route_offsets, member_routes, route_token, route_weights, w1, w2
        ),
        warmup=5,
        repetitions=20,
    )
    upstream_p50, upstream_p95 = benchmark(
        lambda: _triton_moe_baseline(
            baseline,
            x,
            route_offsets,
            member_routes,
            route_token,
            route_weights,
            w1,
            w2,
        ),
        warmup=5,
        repetitions=20,
    )
    print_artifact(artifact, "Triton")
    print(
        "Triton MoE numerical comparison: PASS "
        f"(T={TOKENS}, D={HIDDEN}, F={INTERMEDIATE}, E={EXPERTS}, top_k={TOP_K}, "
        f"generated/reference={generated_error}, upstream/reference={upstream_error})"
    )
    print(
        "Triton MoE end-to-end performance: "
        f"upstream_p50={upstream_p50:.4f} ms, upstream_p95={upstream_p95:.4f} ms, "
        f"generated_p50={generated_p50:.4f} ms, generated_p95={generated_p95:.4f} ms, "
        f"generated/upstream_p50={generated_p50 / upstream_p50:.4f}x"
    )


RUNNERS = {
    "attention": _run_attention,
    "gemm": _run_gemm,
    "moe": _run_moe,
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
