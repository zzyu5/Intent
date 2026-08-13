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


def _load_softmax_baseline(source_path: Path, rows: int, columns: int):
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    tree.body = [node for node in tree.body if node.end_lineno <= 45]
    namespace = {
        "__file__": str(source_path),
        "__name__": "intent_upstream_tilelang_softmax",
        "M": rows,
        "N": columns,
    }
    exec(compile(tree, str(source_path), "exec"), namespace)
    return namespace["softmax_kernel"].compile(
        BLOCK_M=1,
        BLOCK_N=columns,
        dtype=T.float32,
    )


def _load_extended_upstream(kernel: str, source_path: Path):
    if kernel == "w4a8_packed":
        source = _load_module(source_path, "intent_upstream_tilelang_w4a8")
        state = {}

        def run(arguments):
            activation, packed = arguments
            shape = (activation.shape[0], packed.shape[0], activation.shape[1])
            if shape not in state:
                compiled = source.matmul_int8xint4(
                    *shape,
                    T.int8,
                    T.int32,
                    T.int32,
                    num_bits=4,
                    block_M=32,
                    block_N=32,
                    block_K=128,
                    num_stages=2,
                    threads=128,
                )
                state[shape] = (
                    compiled,
                    torch.empty(
                        (shape[1], shape[0]),
                        device=activation.device,
                        dtype=torch.int32,
                    ),
                )
            compiled, output = state[shape]
            compiled.adapter._get_executable()(activation, packed, output)
            return output

        return run
    if kernel == "fp8_gemm":
        source = _load_module(source_path, "intent_upstream_tilelang_fp8")
        state = {}

        def run(arguments):
            lhs, rhs = arguments
            label = "e5m2" if lhs.dtype == torch.float8_e5m2 else "e4m3"
            shape = (lhs.shape[0], rhs.shape[0], lhs.shape[1], label)
            if shape not in state:
                dtype = (
                    source.determine_fp8_type("e5m2")
                    if label == "e5m2"
                    else source.determine_fp8_type()
                )
                compiled = source.matmul.compile(
                    M=shape[0],
                    N=shape[1],
                    K=shape[2],
                    block_M=128,
                    block_N=128,
                    block_K=64,
                    dtype=dtype,
                )
                state[shape] = (
                    compiled,
                    torch.empty(
                        (shape[0], shape[1]),
                        device=lhs.device,
                        dtype=lhs.dtype,
                    ),
                )
            compiled, output = state[shape]
            compiled.adapter._get_executable()(lhs, rhs, output)
            return output

        return run
    if kernel == "bf16_gemm":
        source = _load_module(source_path, "intent_upstream_tilelang_bf16_gemm")
        compiled = {}

        def run(arguments):
            a, b = arguments
            shape = (a.shape[0], b.shape[1], a.shape[1])
            if shape not in compiled:
                compiled[shape] = source.matmul.compile(
                    M=shape[0],
                    N=shape[1],
                    K=shape[2],
                    block_M=128,
                    block_N=128,
                    block_K=32,
                    dtype=T.bfloat16,
                )
            return compiled[shape](a, b)

        return run
    if kernel in {"varlen_attention", "varlen_gqa_prefill"}:
        sys.path.insert(0, str(source_path.parent))
        source = _load_module(
            source_path, "intent_upstream_tilelang_varlen_attention"
        )
        prepared = {}

        def run(arguments):
            q, k, v, lengths, cu_seqlens, _ = arguments
            if not prepared:
                prepared["max_sequence_length"] = int(lengths.max().item())
                query_heads = 1 if q.ndim == 2 else q.shape[1]
                key_value_heads = 1 if k.ndim == 2 else k.shape[1]
                groups = query_heads // key_value_heads
                prepared["compiled"] = source.flashattn(
                    lengths.numel(),
                    groups,
                    q.shape[0],
                    k.shape[0],
                    query_heads,
                    q.shape[-1],
                    True,
                    block_M=64,
                    block_N=64,
                    num_stages=2,
                    threads=128,
                )
            output = prepared["compiled"](
                q[:, None, :] if q.ndim == 2 else q,
                k[:, None, :] if k.ndim == 2 else k,
                v[:, None, :] if v.ndim == 2 else v,
                cu_seqlens,
                cu_seqlens,
                prepared["max_sequence_length"],
            )
            return output[:, 0, :] if q.ndim == 2 else output

        return run
    if kernel == "rms_norm":
        source = _load_module(source_path, "intent_upstream_tilelang_rms_norm")
        compiled = {}

        def run(arguments):
            x, weight, _, _ = arguments
            shape = tuple(x.shape)
            if shape not in compiled:
                compiled[shape] = source.rms_norm.compile(
                    M=shape[0], N=shape[1], blk_m=1
                )
            return compiled[shape](x) * weight

        return run
    if kernel == "dual_gemm":
        source = _load_module(source_path, "intent_upstream_tilelang_dual_gemm")
        compiled = {}

        def run(arguments):
            x, gate_weight, value_weight = arguments
            shape = (x.shape[0], gate_weight.shape[1], x.shape[1])
            if shape not in compiled:
                compiled[shape] = source.matmul.compile(
                    M=shape[0],
                    N=shape[1],
                    K=shape[2],
                    block_M=128,
                    block_N=128,
                    block_K=32,
                )
            matmul = compiled[shape]
            return (
                torch.relu(matmul(x, gate_weight).float())
                * matmul(x, value_weight).float()
            ).half()

        return run
    if kernel == "grouped_gemm":
        grouped = _load_module(
            source_path, "intent_upstream_tilelang_grouped_gemm"
        ).grouped_gemm

        def run(arguments):
            x, offsets, weight = arguments
            batch_sizes = offsets[1:] - offsets[:-1]
            batch_offsets = offsets[:-1].contiguous()
            batch_sizes_list = tuple(int(size) for size in batch_sizes.tolist())
            padded_sizes = [
                math.ceil(size / 128) * 128 for size in batch_sizes_list
            ]
            padded_offsets = [0]
            for size in padded_sizes[:-1]:
                padded_offsets.append(padded_offsets[-1] + size)
            batch_padded_offsets = torch.tensor(
                padded_offsets, device=x.device, dtype=torch.int32
            )
            values = grouped(
                x,
                weight,
                batch_sizes,
                batch_offsets,
                batch_padded_offsets,
                batch_sizes_list,
                128,
                128,
                32,
                False,
                2,
                256,
            )
            return values.float()

        return run
    if kernel == "online_softmax":
        compiled = {}

        def run(arguments):
            x = arguments[0]
            shape = tuple(x.shape)
            if shape not in compiled:
                compiled[shape] = _load_softmax_baseline(
                    source_path, shape[0], shape[1]
                )
            return compiled[shape](x)

        return run
    raise NotImplementedError(f"no TileLang upstream adapter for {kernel}")


def _run_softmax(compiler: str, baseline_source: Path) -> None:
    baseline = _load_softmax_baseline(baseline_source, ROWS, COLUMNS)
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
    generated_p50, generated_p95 = benchmark(
        prepare_kernel_call(artifact, (x,), generated), cuda_graph=True
    )
    upstream_p50, upstream_p95 = benchmark(lambda: baseline(x), cuda_graph=True)
    print_artifact(artifact, "TileLang")
    print(
        "TileLang softmax numerical comparison: PASS "
        f"(shape=({ROWS}, {COLUMNS}), dtype=f32, "
        f"generated/reference={generated_error}, upstream/reference={upstream_error}, "
        f"generated/upstream={generated_upstream_error})"
    )
    print(
        "TileLang softmax kernel-only performance (CUDA Graph): "
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
    generated_p50, generated_p95 = benchmark(
        prepare_kernel_call(artifact, (a, b), generated), cuda_graph=True
    )
    upstream_p50, upstream_p95 = benchmark(
        lambda: baseline(a, b), cuda_graph=True
    )
    print_artifact(artifact, "TileLang")
    print(
        "TileLang GEMM numerical comparison: PASS "
        f"(shape=({M}, {K}) x ({K}, {N}), dtype=f16, "
        f"generated/reference={generated_error}, upstream/reference={upstream_error}, "
        f"generated/upstream={generated_upstream_error})"
    )
    print(
        "TileLang GEMM kernel-only performance (CUDA Graph): "
        f"upstream_p50={upstream_p50:.4f} ms, upstream_p95={upstream_p95:.4f} ms, "
        f"generated_p50={generated_p50:.4f} ms, generated_p95={generated_p95:.4f} ms, "
        f"generated/upstream_p50={generated_p50 / upstream_p50:.4f}x"
    )
    run_gemm_tail_case(artifact, "TileLang")


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
        prepare_kernel_call(artifact, (q, k, v, SCALE), generated),
        cuda_graph=True,
    )
    upstream_p50, upstream_p95 = benchmark(
        lambda: baseline(q_bshd, k_bshd, v_bshd),
        cuda_graph=True,
    )
    print_artifact(artifact, "TileLang")
    print(
        "TileLang attention numerical comparison: PASS "
        f"(shape={shape}, dtype=f16, causal=False, "
        f"generated/reference={generated_error}, upstream/reference={upstream_error}, "
        f"generated/upstream={generated_upstream_error})"
    )
    print(
        "TileLang attention kernel-only performance (CUDA Graph): "
        f"upstream_p50={upstream_p50:.4f} ms, upstream_p95={upstream_p95:.4f} ms, "
        f"generated_p50={generated_p50:.4f} ms, generated_p95={generated_p95:.4f} ms, "
        f"generated/upstream_p50={generated_p50 / upstream_p50:.4f}x"
    )
    run_attention_shape_cases(artifact, "TileLang")
    run_wide_attention_index_case(artifact, "TileLang")


def _tilelang_moe_baseline(
    grouped_gemm,
    x,
    member_routes,
    route_token,
    route_weights,
    w1,
    w2,
    batch_sizes,
    batch_offsets,
    batch_padded_offsets,
    batch_sizes_list,
):
    sorted_routes = member_routes.long()
    sorted_tokens = route_token[sorted_routes].long()
    routed_x = x[sorted_tokens]
    hidden = grouped_gemm(
        routed_x,
        w1,
        batch_sizes,
        batch_offsets,
        batch_padded_offsets,
        batch_sizes_list,
        128,
        128,
        32,
        False,
        2,
        256,
    )
    route_output = grouped_gemm(
        torch.relu(hidden),
        w2,
        batch_sizes,
        batch_offsets,
        batch_padded_offsets,
        batch_sizes_list,
        128,
        128,
        32,
        False,
        2,
        256,
    )
    merged = torch.zeros((TOKENS, HIDDEN), device=x.device, dtype=torch.float32)
    merged.index_add_(
        0,
        sorted_tokens,
        route_weights[sorted_routes, None] * route_output.float(),
    )
    return merged


def _run_moe(compiler: str, baseline_source: Path) -> None:
    upstream = _load_module(baseline_source, "intent_upstream_tilelang_moe")
    artifact = intent.compile(
        moe_expert_ffn,
        target=intent.TileLangTarget(device=0),
        compiler=compiler,
    )
    device = torch.device("cuda", 0)
    route_offsets, member_routes, route_token, route_weights, _ = make_moe_routes(
        device
    )
    batch_sizes = route_offsets[1:] - route_offsets[:-1]
    batch_offsets = route_offsets[:-1].contiguous()
    batch_sizes_list = tuple(int(size) for size in batch_sizes.tolist())
    padded_sizes = [math.ceil(size / 128) * 128 for size in batch_sizes_list]
    padded_offsets = [0]
    for size in padded_sizes[:-1]:
        padded_offsets.append(padded_offsets[-1] + size)
    batch_padded_offsets = torch.tensor(
        padded_offsets,
        device=device,
        dtype=torch.int32,
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
    baseline = lambda: _tilelang_moe_baseline(
        upstream.grouped_gemm,
        x,
        member_routes,
        route_token,
        route_weights,
        w1,
        w2,
        batch_sizes,
        batch_offsets,
        batch_padded_offsets,
        batch_sizes_list,
    )
    upstream_result = baseline()
    reference = moe_reference(
        x, route_offsets, member_routes, route_token, route_weights, w1, w2
    )
    torch.cuda.synchronize()
    generated_error = (generated - reference).abs().max().item()
    upstream_error = (upstream_result - reference).abs().max().item()
    generated_upstream_error = (generated - upstream_result).abs().max().item()
    if generated_error > 2.0e-3 or upstream_error > 2.0e-3:
        raise RuntimeError(
            "TileLang MoE numerical comparison failed: "
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
        baseline,
        warmup=5,
        repetitions=20,
    )
    print_artifact(artifact, "TileLang")
    print(
        "TileLang MoE numerical comparison: PASS "
        f"(T={TOKENS}, D={HIDDEN}, F={INTERMEDIATE}, E={EXPERTS}, top_k={TOP_K}, "
        f"generated/reference={generated_error}, upstream/reference={upstream_error}, "
        f"generated/upstream={generated_upstream_error})"
    )
    print(
        "TileLang MoE end-to-end performance: "
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
            intent.TileLangTarget(device=0),
            "TileLang",
            upstream,
        )
    else:
        if arguments.baseline_source is not None:
            parser.error("unfamiliar programs do not accept an upstream adapter")
        run_unfamiliar(
            arguments.kernel,
            arguments.compiler,
            intent.TileLangTarget(device=0),
            "TileLang",
        )


if __name__ == "__main__":
    main()
