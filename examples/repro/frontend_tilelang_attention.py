from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import intent

from frontend_attention import BATCH
from frontend_attention import HEAD_DIMENSION
from frontend_attention import HEADS
from frontend_attention import SEQUENCE
from frontend_softmax import flash_attention_fwd


SCALE = 1.0 / math.sqrt(HEAD_DIMENSION)


def _load_baseline(source_path: Path):
    spec = importlib.util.spec_from_file_location(
        "intent_upstream_tilelang_attention", source_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.flashattn(
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
        flash_attention_fwd,
        constexprs={"CAUSAL": False},
        target=intent.TileLangTarget(device=0),
        realizer=arguments.intent_realize,
        translator=arguments.intent_translate,
    )
    baseline = _load_baseline(arguments.baseline_source)
    shape = (BATCH, HEADS, SEQUENCE, HEAD_DIMENSION)
    q = torch.randn(shape, device=device, dtype=torch.float16) * 0.5
    k = torch.randn(shape, device=device, dtype=torch.float16) * 0.5
    v = torch.randn(shape, device=device, dtype=torch.float16) * 0.5
    q_bshd = q.permute(0, 2, 1, 3).contiguous()
    k_bshd = k.permute(0, 2, 1, 3).contiguous()
    v_bshd = v.permute(0, 2, 1, 3).contiguous()
    generated = artifact.run(q, k, v, SCALE)
    original_bshd = baseline(q_bshd, k_bshd, v_bshd)
    original = original_bshd.permute(0, 2, 1, 3)
    reference = F.scaled_dot_product_attention(
        q, k, v, is_causal=False, scale=SCALE
    )
    torch.cuda.synchronize()

    generated_error = (generated - reference).abs().max().item()
    original_error = (original - reference).abs().max().item()
    generated_original_error = (generated - original).abs().max().item()
    if generated_error > 2.0e-2 or original_error > 2.0e-2:
        raise RuntimeError(
            "TileLang attention numerical comparison failed: "
            f"generated/reference={generated_error}, "
            f"upstream/reference={original_error}"
        )

    generated_p50, generated_p95 = _bench(
        lambda: artifact.run(q, k, v, SCALE)
    )
    original_p50, original_p95 = _bench(
        lambda: baseline(q_bshd, k_bshd, v_bshd)
    )
    print("=== Intent Kernel IR + TileLang Plan MLIR ===")
    print(artifact.mlir, end="")
    print("=== Generated TileLang source ===")
    print(artifact.source, end="")
    print(
        "TileLang attention numerical comparison: PASS "
        f"(shape={shape}, dtype=f16, causal=False, "
        f"generated/reference={generated_error}, upstream/reference={original_error}, "
        f"generated/upstream={generated_original_error})"
    )
    print(
        "TileLang attention wrapper performance: "
        f"upstream_p50={original_p50:.4f} ms, upstream_p95={original_p95:.4f} ms, "
        f"generated_p50={generated_p50:.4f} ms, generated_p95={generated_p95:.4f} ms, "
        f"generated/upstream_p50={generated_p50 / original_p50:.4f}x"
    )


if __name__ == "__main__":
    main()
