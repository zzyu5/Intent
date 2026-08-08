from __future__ import annotations

import argparse
import ast
import importlib.util
import math
import sys
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


def _load_module(source_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_softmax(compiler: str, baseline_source: Path) -> None:
    utils_path = baseline_source.parents[2] / "support" / "utils.py"
    _load_module(utils_path, "tilegym.ops.cutile.utils")
    baseline = _load_module(
        baseline_source, "tilegym.ops.cutile._intent_baseline_softmax"
    ).softmax
    artifact = intent.compile(
        stable_softmax,
        target=intent.CuTileTarget(device=0),
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
    if generated_error > 1.0e-6 or generated_upstream_error > 1.0e-6:
        raise RuntimeError(
            "cuTile softmax numerical comparison failed: "
            f"generated/reference={generated_error}, "
            f"generated/upstream={generated_upstream_error}"
        )
    upstream_p50, upstream_p95 = benchmark(lambda: baseline(x))
    generated_p50, generated_p95 = benchmark(lambda: artifact.run(x))
    if generated_p50 > upstream_p50 * 1.05:
        raise RuntimeError(
            "cuTile softmax performance regressed by more than 5%: "
            f"generated={generated_p50:.4f} ms, upstream={upstream_p50:.4f} ms"
        )
    print_artifact(artifact, "cuTile")
    print(
        "cuTile softmax numerical comparison: PASS "
        f"(shape=({ROWS}, {COLUMNS}), dtype=f32, "
        f"generated/reference={generated_error}, upstream/reference={upstream_error}, "
        f"generated/upstream={generated_upstream_error})"
    )
    print(
        "cuTile softmax wrapper performance: "
        f"upstream_p50={upstream_p50:.4f} ms, upstream_p95={upstream_p95:.4f} ms, "
        f"generated_p50={generated_p50:.4f} ms, generated_p95={generated_p95:.4f} ms, "
        f"generated/upstream_p50={generated_p50 / upstream_p50:.4f}x"
    )


def _run_gemm(compiler: str, baseline_source: Path) -> None:
    baseline = _load_module(baseline_source, "intent_upstream_cutile_gemm").matmul
    artifact = intent.compile(
        gemm,
        constexprs={"ACTIVATION": Activation.NONE},
        target=intent.CuTileTarget(device=0),
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
    if generated_error > 2.0e-2 or generated_upstream_error > 2.0e-2:
        raise RuntimeError(
            "cuTile GEMM numerical comparison failed: "
            f"generated/reference={generated_error}, "
            f"generated/upstream={generated_upstream_error}"
        )
    upstream_p50, upstream_p95 = benchmark(lambda: baseline(a, b))
    generated_p50, generated_p95 = benchmark(lambda: artifact.run(a, b))
    if generated_p50 > upstream_p50 * 1.05:
        raise RuntimeError(
            "cuTile GEMM performance regressed by more than 5%: "
            f"generated={generated_p50:.4f} ms, upstream={upstream_p50:.4f} ms"
        )
    print_artifact(artifact, "cuTile")
    print(
        "cuTile GEMM numerical comparison: PASS "
        f"(shape=({M}, {K}) x ({K}, {N}), dtype=f16, "
        f"generated/reference={generated_error}, upstream/reference={upstream_error}, "
        f"generated/upstream={generated_upstream_error})"
    )
    print(
        "cuTile GEMM wrapper performance: "
        f"upstream_p50={upstream_p50:.4f} ms, upstream_p95={upstream_p95:.4f} ms, "
        f"generated_p50={generated_p50:.4f} ms, generated_p95={generated_p95:.4f} ms, "
        f"generated/upstream_p50={generated_p50 / upstream_p50:.4f}x"
    )


def _load_attention(source_path: Path):
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    tree.body = [
        node
        for node in tree.body
        if node.end_lineno <= 210
        and not (
            isinstance(node, ast.ImportFrom) and node.module == "utils.benchmark"
        )
    ]
    namespace = {
        "__file__": str(source_path),
        "__name__": "intent_upstream_cutile_attention",
    }
    exec(compile(tree, str(source_path), "exec"), namespace)
    return namespace["cutile_fmha"]


def _run_attention(compiler: str, baseline_source: Path) -> None:
    baseline = _load_attention(baseline_source)
    artifact = intent.compile(
        flash_attention_fwd,
        constexprs={"CAUSAL": False},
        target=intent.CuTileTarget(device=0),
        compiler=compiler,
    )
    shape = (BATCH, HEADS, SEQUENCE, HEAD_DIMENSION)
    q = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.5
    k = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.5
    v = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.5
    generated = artifact.run(q, k, v, SCALE)
    upstream = baseline(
        q,
        k,
        v,
        qk_scale=SCALE,
        tile_m=128,
        tile_n=128,
        query_group_size=1,
        causal=False,
    )
    reference = F.scaled_dot_product_attention(
        q, k, v, is_causal=False, scale=SCALE
    )
    torch.cuda.synchronize()
    generated_error = (generated - reference).abs().max().item()
    upstream_error = (upstream - reference).abs().max().item()
    generated_upstream_error = (generated - upstream).abs().max().item()
    if generated_error > 2.0e-2 or generated_upstream_error > 2.0e-2:
        raise RuntimeError(
            "cuTile attention numerical comparison failed: "
            f"generated/reference={generated_error}, "
            f"generated/upstream={generated_upstream_error}"
        )
    upstream_p50, upstream_p95 = benchmark(
        lambda: baseline(
            q,
            k,
            v,
            qk_scale=SCALE,
            tile_m=128,
            tile_n=128,
            query_group_size=1,
            causal=False,
        )
    )
    generated_p50, generated_p95 = benchmark(
        lambda: artifact.run(q, k, v, SCALE)
    )
    print_artifact(artifact, "cuTile")
    print(
        "cuTile attention numerical comparison: PASS "
        f"(shape={shape}, dtype=f16, causal=False, "
        f"generated/reference={generated_error}, upstream/reference={upstream_error}, "
        f"generated/upstream={generated_upstream_error})"
    )
    print(
        "cuTile attention wrapper performance: "
        f"upstream_p50={upstream_p50:.4f} ms, upstream_p95={upstream_p95:.4f} ms, "
        f"generated_p50={generated_p50:.4f} ms, generated_p95={generated_p95:.4f} ms, "
        f"generated/upstream_p50={generated_p50 / upstream_p50:.4f}x"
    )


def _cutile_moe_baseline(
    upstream,
    x,
    route_weights,
    w1_upstream,
    w2_upstream,
    sorted_routes,
    sorted_experts,
):
    tile_m = 128
    tile_n = 64
    tile_k = 64
    hidden = torch.empty(
        (TOKENS, TOP_K, INTERMEDIATE), device=x.device, dtype=torch.float16
    )
    upstream.invoke_fused_moe_kernel(
        x,
        w1_upstream,
        hidden,
        route_weights.reshape(TOKENS, TOP_K),
        sorted_routes,
        sorted_experts,
        mul_routed_weight=False,
        num_token_replicas=TOP_K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
    )
    routed_output = torch.empty(
        (TOKENS, TOP_K, HIDDEN), device=x.device, dtype=torch.float16
    )
    upstream.invoke_fused_moe_kernel(
        torch.relu(hidden).reshape(TOKENS * TOP_K, INTERMEDIATE),
        w2_upstream,
        routed_output,
        route_weights.reshape(TOKENS, TOP_K),
        sorted_routes,
        sorted_experts,
        mul_routed_weight=True,
        num_token_replicas=1,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
    )
    return routed_output.float().sum(dim=1)


def _run_moe(compiler: str, baseline_source: Path) -> None:
    upstream_module = _load_module(baseline_source, "intent_upstream_cutile_moe")
    artifact = intent.compile(
        moe_expert_ffn,
        target=intent.CuTileTarget(device=0),
        compiler=compiler,
    )
    device = torch.device("cuda", 0)
    route_offsets, member_routes, route_token, route_weights, topk_ids = (
        make_moe_routes(device)
    )
    sorted_routes, sorted_experts = upstream_module.moe_align_tile_size_torch(
        topk_ids, 128, EXPERTS
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
    w1_upstream = w1.transpose(1, 2).contiguous()
    w2_upstream = w2.transpose(1, 2).contiguous()
    generated = artifact.run(
        x, route_offsets, member_routes, route_token, route_weights, w1, w2
    )
    upstream = _cutile_moe_baseline(
        upstream_module,
        x,
        route_weights,
        w1_upstream,
        w2_upstream,
        sorted_routes,
        sorted_experts,
    )
    reference = moe_reference(
        x, route_offsets, member_routes, route_token, route_weights, w1, w2
    )
    torch.cuda.synchronize()
    generated_error = (generated - reference).abs().max().item()
    upstream_error = (upstream - reference).abs().max().item()
    if generated_error > 2.0e-3 or upstream_error > 2.0e-3:
        raise RuntimeError(
            "cuTile MoE numerical comparison failed: "
            f"generated/reference={generated_error}, "
            f"upstream/reference={upstream_error}"
        )
    generated_p50, generated_p95 = benchmark(
        lambda: artifact.run(
            x, route_offsets, member_routes, route_token, route_weights, w1, w2
        ),
        warmup=5,
        repetitions=20,
    )
    upstream_p50, upstream_p95 = benchmark(
        lambda: _cutile_moe_baseline(
            upstream_module,
            x,
            route_weights,
            w1_upstream,
            w2_upstream,
            sorted_routes,
            sorted_experts,
        ),
        warmup=5,
        repetitions=20,
    )
    print_artifact(artifact, "cuTile")
    print(
        "cuTile MoE numerical comparison: PASS "
        f"(T={TOKENS}, D={HIDDEN}, F={INTERMEDIATE}, E={EXPERTS}, top_k={TOP_K}, "
        f"generated/reference={generated_error}, upstream/reference={upstream_error})"
    )
    print(
        "cuTile MoE end-to-end performance: "
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
