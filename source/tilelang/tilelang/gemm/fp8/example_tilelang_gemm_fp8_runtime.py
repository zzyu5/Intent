import importlib.util
from pathlib import Path

import torch


def load_source():
    source_path = Path(__file__).with_name("example_tilelang_gemm_fp8.py")
    spec = importlib.util.spec_from_file_location("tilelang_fp8_gemm_source", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    m, k, n = 4096, 4096, 14336
    source = load_source()
    dtype = source.determine_fp8_type()
    torch_dtype = source.T.dtype(dtype).as_torch()
    a = torch.randn((m, k), device="cuda", dtype=torch.float16).to(torch_dtype)
    b = torch.randn((n, k), device="cuda", dtype=torch.float16).to(torch_dtype)
    kernel = source.matmul.compile(
        M=m,
        N=n,
        K=k,
        block_M=128,
        block_N=128,
        block_K=64,
        dtype=dtype,
    )

    kernel(a, b)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = kernel(a, b)
    end.record()
    torch.cuda.synchronize()

    print(f"A: shape={tuple(a.shape)}, dtype={a.dtype}")
    print(f"B^T storage: shape={tuple(b.shape)}, dtype={b.dtype}")
    print(f"output: shape={tuple(output.shape)}, dtype={output.dtype}, mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
