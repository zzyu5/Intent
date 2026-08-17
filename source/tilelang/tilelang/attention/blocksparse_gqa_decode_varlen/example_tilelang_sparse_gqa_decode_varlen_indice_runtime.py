import importlib.util
import sys
from pathlib import Path

import torch


SOURCE = Path(__file__).with_name("example_tilelang_sparse_gqa_decode_varlen_indice.py")
sys.path.insert(0, str(SOURCE.parent))
spec = importlib.util.spec_from_file_location("local_tilelang_sparse_gqa_varlen", SOURCE)
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
STATE = {}


def upstream(arguments):
    q, k, v, block_indices, cache_seqlens, split_offsets, scale = arguments
    expected_scale = q.shape[-1] ** -0.5
    if abs(float(scale) - expected_scale) > 1.0e-12:
        raise ValueError("TileLang block-sparse source uses head-dimension scaling")
    splits = split_offsets.numel() - 1
    key = (
        tuple(q.shape),
        q.dtype,
        q.device,
        tuple(k.shape),
        k.dtype,
        k.device,
        tuple(v.shape),
        v.dtype,
        v.device,
        tuple(block_indices.shape),
        block_indices.dtype,
        block_indices.device,
        splits,
    )
    if key not in STATE:
        partial_lse = torch.empty(
            (q.shape[0], q.shape[1], splits),
            device=q.device,
            dtype=torch.float32,
        )
        partial_output = torch.empty(
            (q.shape[0], q.shape[1], splits, v.shape[-1]),
            device=q.device,
            dtype=torch.float32,
        )
        kernel = source.flashattn(
            q.shape[0],
            q.shape[1],
            k.shape[2],
            q.shape[2],
            v.shape[3],
            block_N=64,
            block_H=64,
            num_stages=2,
            threads=128,
        )
        output = kernel(
            q,
            k,
            v,
            block_indices,
            cache_seqlens,
            partial_lse,
            partial_output,
        )
        STATE[key] = (
            kernel.adapter._get_executable(),
            partial_lse,
            partial_output,
            output,
        )
        return output
    executable, partial_lse, partial_output, output = STATE[key]
    executable(
        q,
        k,
        v,
        block_indices,
        cache_seqlens,
        partial_lse,
        partial_output,
        output,
    )
    return output


def main():
    batch, q_heads, kv_heads = 8, 32, 8
    max_cache_length, head_dim, block_size = 8192, 128, 32
    selected_blocks = 128
    q = torch.randn(batch, q_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, max_cache_length, kv_heads, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    selected = torch.arange(
        max_cache_length // block_size - 1,
        max_cache_length // block_size - selected_blocks - 1,
        -1,
        device="cuda",
        dtype=torch.int32,
    )
    block_indices = selected.view(1, 1, -1).expand(batch, kv_heads, -1).contiguous()
    cache_seqlens = torch.full((batch,), max_cache_length, device="cuda", dtype=torch.int32)

    output = source.sparse_gqa_decode_varlen_indice(
        q, k, v, block_indices, cache_seqlens, max_cache_length, block_size
    )
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = source.sparse_gqa_decode_varlen_indice(
        q, k, v, block_indices, cache_seqlens, max_cache_length, block_size
    )
    end.record()
    torch.cuda.synchronize()

    print(
        f"q={tuple(q.shape)} k={tuple(k.shape)} selected_blocks={tuple(block_indices.shape)} "
        f"cache_length={max_cache_length}"
    )
    print(f"output={tuple(output.shape)} mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
