import importlib.util
from pathlib import Path

import torch


def load_source():
    source_path = Path(__file__).with_name("BlockScaledMatMul.py")
    spec = importlib.util.spec_from_file_location("cutile_block_scaled_matmul_source", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    m, k, n = 4096, 4096, 14336
    source = load_source()
    a, a_scale = source.block_quantize(
        torch.randn((m, k), device="cuda", dtype=torch.float32), 32
    )
    b_rows, b_scale_rows = source.block_quantize(
        torch.randn((n, k), device="cuda", dtype=torch.float32), 32
    )
    b = b_rows.T
    b_scale = b_scale_rows.T

    source.cutile_block_scaled_matmul(a, a_scale, b, b_scale)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = source.cutile_block_scaled_matmul(a, a_scale, b, b_scale)
    end.record()
    torch.cuda.synchronize()

    print(f"A: shape={tuple(a.shape)}, dtype={a.dtype}, scales={tuple(a_scale.shape)}")
    print(f"B: shape={tuple(b.shape)}, dtype={b.dtype}, scales={tuple(b_scale.shape)}")
    print(f"output: shape={tuple(output.shape)}, dtype={output.dtype}, mean={output.mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
