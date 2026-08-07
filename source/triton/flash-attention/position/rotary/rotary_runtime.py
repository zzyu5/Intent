import importlib.util
from pathlib import Path

import torch


SOURCE = Path(__file__).with_name("rotary.py")
spec = importlib.util.spec_from_file_location("local_flash_attn_rotary", SOURCE)
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def main():
    batch, seqlen, heads, head_dim = 4, 4096, 32, 128
    x = torch.randn(batch, seqlen, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(seqlen, device="cuda", dtype=torch.float32)
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2, device="cuda", dtype=torch.float32) / head_dim)
    )
    angles = torch.outer(positions, inv_freq)
    cos, sin = angles.cos(), angles.sin()

    output = source.apply_rotary(x, cos, sin)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = source.apply_rotary(x, cos, sin)
    end.record()
    torch.cuda.synchronize()

    print(f"x={tuple(x.shape)} cos={tuple(cos.shape)} sin={tuple(sin.shape)} dtype={x.dtype}")
    print(f"output={tuple(output.shape)} mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
