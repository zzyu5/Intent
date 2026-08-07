import importlib.util
from pathlib import Path

import torch


def load_source():
    source_path = Path(__file__).with_name("rms_norm.py")
    spec = importlib.util.spec_from_file_location("tilelang_rms_norm_source", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    tokens, hidden = 8192, 4096
    x = torch.randn((tokens, hidden), device="cuda", dtype=torch.float32)
    source = load_source()
    kernel = source.rms_norm.compile(M=tokens, N=hidden, blk_m=1)

    kernel(x)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = kernel(x)
    end.record()
    torch.cuda.synchronize()

    print(f"input: shape={tuple(x.shape)}, dtype={x.dtype}")
    print(f"output: shape={tuple(output.shape)}, dtype={output.dtype}, mean={output.mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
