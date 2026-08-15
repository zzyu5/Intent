from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import intent
from kernels.streaming.attention import BATCH
from kernels.streaming.attention import HEAD_DIMENSION
from kernels.streaming.attention import HEADS
from kernels.streaming.attention import SCALE
from kernels.streaming.attention import SEQUENCE
from kernels.streaming.attention import flash_attention_fwd
from kernels.contraction.gemm import Activation
from kernels.contraction.gemm import K
from kernels.contraction.gemm import M
from kernels.contraction.gemm import N
from kernels.contraction.gemm import gemm
from kernels.ragged.moe import EXPERTS
from kernels.ragged.moe import HIDDEN
from kernels.ragged.moe import INTERMEDIATE
from kernels.ragged.moe import TOKENS
from kernels.ragged.moe import TOP_K
from kernels.ragged.moe import moe_expert_ffn
from kernels.normalization.softmax import COLUMNS
from kernels.normalization.softmax import ROWS
from kernels.normalization.softmax import stable_softmax
from repro.common.extended import EXTENDED_RUNNERS
from repro.common.extended import run_attention_shape_cases
from repro.common.extended import run_gemm_tail_case
from repro.common.extended import run_wide_attention_index_case
from repro.common.extended import run_extended
from repro.common.unfamiliar import UNFAMILIAR_RUNNERS
from repro.common.unfamiliar import run_unfamiliar
from repro.common.support import benchmark
from repro.common.support import make_moe_routes
from repro.common.support import moe_reference
from repro.common.support import prepare_kernel_call
from repro.common.support import print_artifact


def _load_module(source_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_extended_upstream(kernel: str, source_path: Path):
    if kernel == "block_scaled_matmul":
        block_scaled = _load_module(
            source_path, "intent_upstream_cutile_block_scaled"
        )
        state = {}

        def run(arguments):
            lhs, lhs_scale, rhs, rhs_scale = arguments
            lhs = lhs.reshape(lhs.shape[0], -1)
            rhs = rhs.reshape(-1, rhs.shape[-1])
            if not state:
                state["output"] = torch.empty(
                    (lhs.shape[0], rhs.shape[1]),
                    device=lhs.device,
                    dtype=torch.float32,
                )
            tm, tn, tk, scaling_block_size = 256, 256, 128, 32
            grid = (
                block_scaled.ct.cdiv(lhs.shape[0], tm)
                * block_scaled.ct.cdiv(rhs.shape[1], tn),
                1,
                1,
            )
            block_scaled.ct.launch(
                torch.cuda.current_stream(),
                grid,
                block_scaled.block_scaled_matmul_kernel,
                (
                    lhs,
                    lhs_scale,
                    rhs,
                    rhs_scale,
                    state["output"],
                    tm,
                    tn,
                    tk,
                    scaling_block_size,
                ),
            )
            return state["output"]

        return run
    if kernel == "splitk_attention_reduce":
        utils_path = source_path.parents[2] / "support" / "utils.py"
        _load_module(utils_path, "tilegym.ops.cutile.utils")
        splitk_reduce = _load_module(
            source_path, "tilegym.ops.cutile._intent_splitk_reduce"
        ).splitk_reduce

        state = {}

        def run(arguments):
            partial, partial_lse = arguments
            if not state:
                state["output"] = torch.empty(
                    partial.shape[0],
                    partial.shape[1],
                    partial.shape[3],
                    device=partial.device,
                    dtype=partial.dtype,
                )
            return splitk_reduce(partial, partial_lse, state["output"], 8192)

        return run
    if kernel == "mla_prefill":
        source = _load_module(source_path, "intent_upstream_cutile_mla_prefill")
        state = {}

        def run(arguments):
            q, qpe, k, kpe, v, scale = arguments
            if not state:
                state["output"] = torch.empty_like(q)
            head_group = q.shape[1] // k.shape[1]
            source._cutile_autotune_mla(
                torch.cuda.current_stream(),
                q,
                qpe,
                k,
                kpe,
                v,
                state["output"],
                scale,
                q.shape[1],
                head_group,
            )
            return state["output"]

        return run
    if kernel == "batched_gemm":
        bmm = _load_module(source_path, "intent_upstream_cutile_batched_gemm").bmm
        return lambda arguments: bmm(
            arguments[0],
            arguments[1],
            transpose_a=arguments[2],
            transpose_b=arguments[3],
            static_persistent=True,
        )
    if kernel == "bf16_gemm":
        matmul = _load_module(source_path, "intent_upstream_cutile_bf16_gemm").matmul
        return lambda arguments: matmul(arguments[0], arguments[1])
    if kernel == "swiglu_forward":
        utils_path = source_path.parents[2] / "support" / "utils.py"
        _load_module(utils_path, "tilegym.ops.cutile.utils")
        silu_and_mul = _load_module(
            source_path, "tilegym.ops.cutile._intent_silu_and_mul"
        ).silu_and_mul
        return lambda arguments: silu_and_mul(arguments[2])
    if kernel == "layer_norm":
        layer_norm = _load_module(
            source_path, "intent_upstream_cutile_layer_norm"
        ).cutile_layer_norm
        return lambda arguments: layer_norm(
            arguments[0], arguments[1], arguments[2], arguments[4]
        )
    if kernel == "dual_gemm":
        matmul = _load_module(
            source_path, "intent_upstream_cutile_dual_gemm"
        ).matmul
        return lambda arguments: (
            torch.relu(matmul(arguments[0], arguments[1]).float())
            * matmul(arguments[0], arguments[2]).float()
        ).half()
    if kernel == "grouped_gemm":
        grouped = _load_module(
            source_path, "intent_upstream_cutile_grouped_gemm"
        ).group_gemm

        def run(arguments):
            x, offsets, weight = arguments
            values = grouped(
                [
                    x[offsets[group] : offsets[group + 1]]
                    for group in range(weight.shape[0])
                ],
                [weight[group] for group in range(weight.shape[0])],
            )
            return torch.cat(
                [group_values.float() for group_values in values], dim=0
            )

        return run
    if kernel == "online_softmax":
        utils_path = source_path.parents[2] / "support" / "utils.py"
        _load_module(utils_path, "tilegym.ops.cutile.utils")
        softmax = _load_module(
            source_path, "tilegym.ops.cutile._intent_online_softmax"
        ).softmax
        return lambda arguments: softmax(arguments[0])
    raise NotImplementedError(f"no cuTile upstream adapter for {kernel}")


def _load_unfamiliar_upstream(kernel: str, source_path: Path):
    if kernel not in {"rope_qk_full", "rope_qk_partial", "rope_qk_inverse"}:
        raise ValueError(f"unfamiliar program '{kernel}' has no cuTile adapter")
    utils_path = source_path.parents[2] / "support" / "utils.py"
    _load_module(utils_path, "tilegym.ops.cutile.utils")
    source = _load_module(source_path, "tilegym.ops.cutile._intent_rope")

    def run(arguments):
        query, key, cosine, sine = arguments
        rotary_dimension = cosine.shape[-1]
        rope_dimension = (
            None if rotary_dimension == query.shape[-1] else rotary_dimension
        )
        output_query, output_key, _, _ = source._rope_forward(
            query, key, cosine, sine, rope_dim=rope_dimension
        )
        return output_query, output_key

    return run


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
    upstream_p50, upstream_p95 = benchmark(lambda: baseline(x), cuda_graph=True)
    generated_p50, generated_p95 = benchmark(
        prepare_kernel_call(artifact, (x,), generated), cuda_graph=True
    )
    print_artifact(artifact, "cuTile")
    print(
        "cuTile softmax numerical comparison: PASS "
        f"(shape=({ROWS}, {COLUMNS}), dtype=f32, "
        f"generated/reference={generated_error}, upstream/reference={upstream_error}, "
        f"generated/upstream={generated_upstream_error})"
    )
    print(
        "cuTile softmax kernel-only performance (CUDA Graph): "
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
    upstream_p50, upstream_p95 = benchmark(
        lambda: baseline(a, b), cuda_graph=True
    )
    generated_p50, generated_p95 = benchmark(
        prepare_kernel_call(artifact, (a, b), generated), cuda_graph=True
    )
    print_artifact(artifact, "cuTile")
    print(
        "cuTile GEMM numerical comparison: PASS "
        f"(shape=({M}, {K}) x ({K}, {N}), dtype=f16, "
        f"generated/reference={generated_error}, upstream/reference={upstream_error}, "
        f"generated/upstream={generated_upstream_error})"
    )
    print(
        "cuTile GEMM kernel-only performance (CUDA Graph): "
        f"upstream_p50={upstream_p50:.4f} ms, upstream_p95={upstream_p95:.4f} ms, "
        f"generated_p50={generated_p50:.4f} ms, generated_p95={generated_p95:.4f} ms, "
        f"generated/upstream_p50={generated_p50 / upstream_p50:.4f}x"
    )
    run_gemm_tail_case(artifact, "cuTile")


def _load_attention(source_path: Path):
    module = _load_module(source_path, "intent_upstream_cutile_attention")
    return module.source.tile_fmha


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
        scaling=SCALE,
        is_causal=False,
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
            scaling=SCALE,
            is_causal=False,
        ),
        cuda_graph=True,
    )
    generated_p50, generated_p95 = benchmark(
        prepare_kernel_call(artifact, (q, k, v, SCALE), generated),
        cuda_graph=True,
    )
    print_artifact(artifact, "cuTile")
    print(
        "cuTile attention numerical comparison: PASS "
        f"(shape={shape}, dtype=f16, causal=False, "
        f"generated/reference={generated_error}, upstream/reference={upstream_error}, "
        f"generated/upstream={generated_upstream_error})"
    )
    print(
        "cuTile attention kernel-only performance (CUDA Graph): "
        f"upstream_p50={upstream_p50:.4f} ms, upstream_p95={upstream_p95:.4f} ms, "
        f"generated_p50={generated_p50:.4f} ms, generated_p95={generated_p95:.4f} ms, "
        f"generated/upstream_p50={generated_p50 / upstream_p50:.4f}x"
    )
    run_attention_shape_cases(artifact, "cuTile")
    run_wide_attention_index_case(artifact, "cuTile")


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
    parser.add_argument(
        "kernel", choices=sorted(RUNNERS | EXTENDED_RUNNERS | UNFAMILIAR_RUNNERS)
    )
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--baseline-source", type=Path)
    arguments = parser.parse_args()
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    if arguments.kernel in RUNNERS:
        if arguments.baseline_source is None:
            parser.error("the selected upstream comparison requires --baseline-source")
        RUNNERS[arguments.kernel](arguments.compiler, arguments.baseline_source)
    elif arguments.kernel in EXTENDED_RUNNERS:
        upstream = (
            _load_extended_upstream(arguments.kernel, arguments.baseline_source)
            if arguments.baseline_source is not None
            else None
        )
        run_extended(
            arguments.kernel,
            arguments.compiler,
            intent.CuTileTarget(device=0),
            "cuTile",
            upstream,
        )
    else:
        upstream = (
            _load_unfamiliar_upstream(arguments.kernel, arguments.baseline_source)
            if arguments.baseline_source is not None
            else None
        )
        run_unfamiliar(
            arguments.kernel,
            arguments.compiler,
            intent.CuTileTarget(device=0),
            "cuTile",
            upstream,
        )


if __name__ == "__main__":
    main()
