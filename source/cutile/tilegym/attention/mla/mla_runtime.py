import importlib.util
import sys
from pathlib import Path

import tilegym
import torch


SOURCE = Path(__file__).with_name("mla.py")
spec = importlib.util.spec_from_file_location("tilegym.ops.cutile._local_mla", SOURCE)
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)


def main():
    batch, q_heads, kv_heads, seqlen = 1, 128, 1, 2048
    head_dim, pe_dim = 128, 64
    q = torch.randn(batch, q_heads, seqlen, head_dim, device="cuda", dtype=torch.bfloat16)
    q_pe = torch.randn(batch, q_heads, seqlen, pe_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, kv_heads, seqlen, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    k_pe = torch.randn(batch, kv_heads, seqlen, pe_dim, device="cuda", dtype=torch.bfloat16)

    output = source.tile_mla(q, k, v, q_pe, k_pe, True, (head_dim + pe_dim) ** -0.5)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = source.tile_mla(q, k, v, q_pe, k_pe, True, (head_dim + pe_dim) ** -0.5)
    end.record()
    torch.cuda.synchronize()

    print(
        f"q={tuple(q.shape)} q_pe={tuple(q_pe.shape)} k={tuple(k.shape)} "
        f"k_pe={tuple(k_pe.shape)} dtype={q.dtype}"
    )
    print(f"output={tuple(output.shape)} mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
