import importlib.util
from pathlib import Path

import torch


def load_source():
    source_path = Path(__file__).with_name("example_mha_fwd_bshd.py")
    spec = importlib.util.spec_from_file_location("tilelang_flash_attention_source", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    batch, sequence, heads, head_dim = 4, 4096, 32, 128
    shape = (batch, sequence, heads, head_dim)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn(shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, device="cuda", dtype=torch.float16)
    source = load_source()
    kernel = source.flashattn(
        batch,
        heads,
        sequence,
        head_dim,
        True,
        block_M=128,
        block_N=128,
        num_stages=1,
        threads=128,
    )

    kernel(q, k, v)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = kernel(q, k, v)
    end.record()
    torch.cuda.synchronize()

    print(f"Q/K/V: shape={shape}, dtype={q.dtype}, layout=BSHD, causal=True")
    print(f"output: shape={tuple(output.shape)}, dtype={output.dtype}, mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
