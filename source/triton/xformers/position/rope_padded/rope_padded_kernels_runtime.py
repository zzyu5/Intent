import importlib.util
import sys
import types
from pathlib import Path

import torch
import triton


def load_source():
    package_root = Path(__file__).parents[2]
    xformers_package = types.ModuleType("xformers")
    xformers_package.__path__ = []
    triton_package = types.ModuleType("xformers.triton")
    triton_package.__path__ = []
    sys.modules["xformers"] = xformers_package
    sys.modules["xformers.triton"] = triton_package

    importing_path = package_root / "support" / "triton" / "importing.py"
    importing_spec = importlib.util.spec_from_file_location(
        "xformers.triton.importing", importing_path
    )
    importing_module = importlib.util.module_from_spec(importing_spec)
    sys.modules[importing_spec.name] = importing_module
    importing_spec.loader.exec_module(importing_module)

    source_path = Path(__file__).with_name("rope_padded_kernels.py")
    source_spec = importlib.util.spec_from_file_location(
        "xformers_rope_padded_source", source_path
    )
    source = importlib.util.module_from_spec(source_spec)
    source_spec.loader.exec_module(source)
    return source


def main():
    batch, q_heads, kv_heads, cache_length, head_dim = 32, 32, 8, 8192, 128
    total_queries = batch
    xq = torch.randn(
        (1, total_queries, q_heads, head_dim), device="cuda", dtype=torch.float16
    )
    xk = torch.randn(
        (1, total_queries, kv_heads, head_dim), device="cuda", dtype=torch.float16
    )
    xv = torch.randn_like(xk)
    out_q = torch.empty_like(xq)
    padded_cache_length = cache_length + 1
    cache_k = torch.empty(
        (1, batch * padded_cache_length, kv_heads, head_dim),
        device="cuda",
        dtype=torch.float16,
    )
    cache_v = torch.empty_like(cache_k)
    seqstart_q = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    seqstart_k = (
        torch.arange(batch + 1, device="cuda", dtype=torch.int32)
        * padded_cache_length
    )
    sequence_lengths = torch.full(
        (batch,), padded_cache_length, device="cuda", dtype=torch.int32
    )
    source = load_source()
    block_size = max(128, min(4096, triton.next_power_of_2(head_dim)))
    q_stride = xq.stride()
    k_stride = xk.stride()
    v_stride = xv.stride()
    cache_k_stride = cache_k.stride()
    cache_v_stride = cache_v.stride()
    out_stride = out_q.stride()
    total_heads = q_heads + 2 * kv_heads

    def launch():
        source._rope_padded_kernel[(1, batch, total_heads)](
            xq,
            xk,
            xv,
            out_q,
            cache_k,
            cache_v,
            seqstart_q,
            seqstart_k,
            sequence_lengths,
            10000.0,
            1.0,
            False,
            0,
            0,
            0,
            0,
            None,
            None,
            q_heads,
            q_heads + kv_heads,
            1,
            head_dim,
            q_stride[1],
            0,
            q_stride[-2],
            k_stride[1],
            0,
            k_stride[-2],
            v_stride[1],
            0,
            v_stride[-2],
            cache_k_stride[1],
            0,
            cache_k_stride[-2],
            cache_v_stride[1],
            0,
            cache_v_stride[-2],
            seqstart_q.stride(0),
            seqstart_k.stride(0),
            sequence_lengths.stride(0),
            out_stride[1],
            0,
            out_stride[-2],
            0,
            "f32",
            const_batch_strides=False,
            cache_padding_length=0,
            seqlenk_shift=0,
            BLOCK_SIZE=block_size,
            adjacents=False,
            num_warps=1,
        )

    launch()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    launch()
    end.record()
    torch.cuda.synchronize()

    print(f"Q: shape={tuple(xq.shape)}, K/V update={tuple(xk.shape)}, dtype={xq.dtype}")
    print(f"KV cache: shape={tuple(cache_k.shape)}, context={cache_length}")
    print(f"Q output mean={out_q.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
