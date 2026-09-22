import importlib.util
import math
from pathlib import Path

import torch
import tilelang


def load_source():
    source_path = Path(__file__).with_name("example_mla_decode_paged.py")
    spec = importlib.util.spec_from_file_location("tilelang_paged_mla_source", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    batch, query_heads, kv_heads = 32, 128, 1
    kv_length, value_dim, rope_dim = 8192, 128, 64
    head_dim = value_dim + rope_dim
    block_size, split_k = 64, 1
    cache_lengths = torch.tensor(
        [kv_length + 2 * index for index in range(batch)], device="cuda", dtype=torch.int32
    )
    max_length = math.ceil(cache_lengths.max().item() / 256) * 256
    block_table = torch.arange(
        batch * max_length // block_size, device="cuda", dtype=torch.int32
    ).view(batch, max_length // block_size)
    q = torch.randn((batch, 1, query_heads, head_dim), device="cuda", dtype=torch.float16)
    blocked_k = torch.randn(
        (block_table.numel(), block_size, kv_heads, head_dim), device="cuda", dtype=torch.float16
    )
    q_nope = q[..., :value_dim].contiguous().view(batch, query_heads, value_dim)
    q_pe = q[..., value_dim:].contiguous().view(batch, query_heads, rope_dim)
    k_nope = blocked_k[..., :value_dim].contiguous().view(-1, kv_heads, value_dim)
    k_pe = blocked_k[..., value_dim:].contiguous().view(-1, kv_heads, rope_dim)
    glse = torch.empty((batch, query_heads, split_k), device="cuda", dtype=torch.float16)
    partial = torch.empty(
        (batch, query_heads, split_k, value_dim), device="cuda", dtype=torch.float16
    )

    source = load_source()
    kernel = source.mla_decode_tilelang(
        batch,
        query_heads,
        kv_heads,
        max_length,
        value_dim,
        rope_dim,
        64,
        64,
        split_k,
        block_size,
        head_dim**-0.5,
    )
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)

    def launch():
        return profiler.func(q_nope, q_pe, k_nope, k_pe, block_table, cache_lengths, glse, partial)

    launch()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = launch()
    end.record()
    torch.cuda.synchronize()

    print(f"Q: shape={tuple(q.shape)}, dtype={q.dtype}")
    print(f"paged KV: shape={tuple(blocked_k.shape)}, block_table={tuple(block_table.shape)}")
    print(f"cache lengths: min={cache_lengths.min().item()}, max={cache_lengths.max().item()}")
    print(f"output: shape={tuple(output.shape)}, dtype={output.dtype}, mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
