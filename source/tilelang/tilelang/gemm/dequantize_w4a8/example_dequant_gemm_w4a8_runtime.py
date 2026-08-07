import importlib.util
from pathlib import Path

import torch


def load_source():
    source_path = Path(__file__).with_name("example_dequant_gemm_w4a8.py")
    spec = importlib.util.spec_from_file_location("tilelang_w4a8_gemm_source", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    m, k, n = 4096, 4096, 14336
    a = torch.randint(-128, 128, (m, k), device="cuda", dtype=torch.int8)
    packed_b = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    source = load_source()
    kernel = source.matmul_int8xint4(
        m,
        n,
        k,
        source.T.int8,
        source.T.int32,
        source.T.int32,
        num_bits=4,
        block_M=128,
        block_N=128,
        block_K=128,
        num_stages=2,
        threads=256,
    )

    kernel(a, packed_b)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = kernel(a, packed_b)
    end.record()
    torch.cuda.synchronize()

    print(f"A8: shape={tuple(a.shape)}, dtype={a.dtype}")
    print(f"packed W4: shape={tuple(packed_b.shape)}, dtype={packed_b.dtype}")
    print(f"output: shape={tuple(output.shape)}, dtype={output.dtype}, mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
