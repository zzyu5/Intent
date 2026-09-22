import importlib.util
from pathlib import Path

import torch


SOURCE = Path(__file__).with_name("example_gqa_decode_varlen_logits.py")
spec = importlib.util.spec_from_file_location("local_tilelang_gqa_decode_varlen_logits", SOURCE)
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def main():
    batch, q_heads, kv_heads, head_dim = 16, 32, 8, 64
    max_seqlen = 8192
    lengths = torch.arange(
        max_seqlen, max_seqlen - batch * 128, -128, device="cuda", dtype=torch.int32
    )
    cu_seqlens = torch.zeros(batch + 1, device="cuda", dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
    total_tokens = int(cu_seqlens[-1].item())
    q = torch.randn(batch, q_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(total_tokens, kv_heads, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    sink = torch.zeros(q_heads, device="cuda", dtype=torch.float32)
    kernel = source.flashattn(
        batch, q_heads, kv_heads, max_seqlen, total_tokens, head_dim, False,
        block_N=64, block_H=64, num_split=1, num_stages=2, threads=128,
    )

    output, block_logits = kernel(q, k, v, cu_seqlens, sink)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output, block_logits = kernel(q, k, v, cu_seqlens, sink)
    end.record()
    torch.cuda.synchronize()

    print(
        f"lengths={lengths.tolist()} q={tuple(q.shape)} packed_kv={tuple(k.shape)} "
        f"sink=disabled"
    )
    print(
        f"output={tuple(output.shape)} block_logits={tuple(block_logits.shape)} "
        f"mean={output.float().mean().item():.6f}"
    )
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
