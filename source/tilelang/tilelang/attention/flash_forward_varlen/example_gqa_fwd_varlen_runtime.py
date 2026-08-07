import importlib.util
import sys
from pathlib import Path

import torch


SOURCE = Path(__file__).with_name("example_gqa_fwd_varlen.py")
sys.path.insert(0, str(SOURCE.parent))
spec = importlib.util.spec_from_file_location("local_tilelang_gqa_varlen", SOURCE)
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def main():
    batch, heads, kv_groups, head_dim = 8, 32, 4, 128
    lengths = torch.tensor(
        [4096, 3968, 3840, 3712, 3584, 3456, 3328, 3200],
        device="cuda",
        dtype=torch.int32,
    )
    cu_seqlens = torch.zeros(batch + 1, device="cuda", dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
    total_tokens = int(cu_seqlens[-1].item())
    max_seqlen = int(lengths.max().item())
    kv_heads = heads // kv_groups
    q = torch.randn(total_tokens, heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(total_tokens, kv_heads, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    kernel = source.flashattn(
        batch, kv_groups, total_tokens, total_tokens, heads, head_dim, True,
        block_M=64, block_N=64, num_stages=2, threads=128,
    )

    output = kernel(q, k, v, cu_seqlens, cu_seqlens, max_seqlen)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = kernel(q, k, v, cu_seqlens, cu_seqlens, max_seqlen)
    end.record()
    torch.cuda.synchronize()

    print(
        f"lengths={lengths.tolist()} q={tuple(q.shape)} k={tuple(k.shape)} "
        f"heads={heads} kv_heads={kv_heads}"
    )
    print(f"output={tuple(output.shape)} mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
