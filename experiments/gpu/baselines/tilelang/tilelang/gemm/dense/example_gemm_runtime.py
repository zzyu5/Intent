import importlib.util
from pathlib import Path

import torch


def load_source():
    source_path = Path(__file__).with_name("example_gemm.py")
    spec = importlib.util.spec_from_file_location("tilelang_gemm_source", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    # Mixtral 8x7B MLP up projection, evaluated on 4096 tokens.
    m, k, n = 4096, 4096, 14336
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    source = load_source()
    kernel = source.matmul.compile(M=m, N=n, K=k, block_M=128, block_N=128, block_K=32)

    kernel(a, b)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = kernel(a, b)
    end.record()
    torch.cuda.synchronize()

    print(f"A: shape={tuple(a.shape)}, dtype={a.dtype}")
    print(f"B: shape={tuple(b.shape)}, dtype={b.dtype}")
    print(f"output: shape={tuple(output.shape)}, dtype={output.dtype}, mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
