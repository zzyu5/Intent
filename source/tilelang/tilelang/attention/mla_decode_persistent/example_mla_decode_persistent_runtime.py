import importlib.util
from pathlib import Path

import torch


SOURCE = Path(__file__).with_name("example_mla_decode_persistent.py")
spec = importlib.util.spec_from_file_location("local_tilelang_mla_persistent", SOURCE)
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def main():
    batch, heads, kv_heads = 8, 64, 1
    context, head_dim, pe_dim, num_split = 16384, 128, 64, 4
    q = torch.randn(batch, heads, head_dim, device="cuda", dtype=torch.float16)
    q_pe = torch.randn(batch, heads, pe_dim, device="cuda", dtype=torch.float16)
    kv = torch.randn(batch, context, kv_heads, head_dim, device="cuda", dtype=torch.float16)
    k_pe = torch.randn(batch, context, kv_heads, pe_dim, device="cuda", dtype=torch.float16)
    glse = torch.empty(batch, heads, num_split, device="cuda", dtype=torch.float16)
    partial = torch.empty(
        batch, heads, num_split, head_dim, device="cuda", dtype=torch.float16
    )
    kernel = source.flashattn(
        batch, heads, kv_heads, context, head_dim, pe_dim, 64, 64, num_split
    )

    output = kernel(q, q_pe, kv, k_pe, glse, partial)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = kernel(q, q_pe, kv, k_pe, glse, partial)
    end.record()
    torch.cuda.synchronize()

    print(
        f"q={tuple(q.shape)} q_pe={tuple(q_pe.shape)} kv={tuple(kv.shape)} "
        f"k_pe={tuple(k_pe.shape)}"
    )
    print(f"output={tuple(output.shape)} mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
