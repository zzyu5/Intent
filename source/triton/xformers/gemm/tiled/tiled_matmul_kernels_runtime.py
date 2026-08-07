import importlib.util
import sys
import types
from pathlib import Path

import torch


def load_source():
    xformers_package = types.ModuleType("xformers")
    xformers_package.__path__ = []
    ops_package = types.ModuleType("xformers.ops")
    ops_package.__path__ = []
    triton_ops_package = types.ModuleType("xformers.ops._triton")
    triton_ops_package.__path__ = []
    sys.modules["xformers"] = xformers_package
    sys.modules["xformers.ops"] = ops_package
    sys.modules["xformers.ops._triton"] = triton_ops_package

    perf_path = Path(__file__).with_name("matmul_perf_model.py")
    perf_spec = importlib.util.spec_from_file_location(
        "xformers.ops._triton.matmul_perf_model", perf_path
    )
    perf_module = importlib.util.module_from_spec(perf_spec)
    sys.modules[perf_spec.name] = perf_module
    perf_spec.loader.exec_module(perf_module)

    source_path = Path(__file__).with_name("tiled_matmul_kernels.py")
    source_spec = importlib.util.spec_from_file_location(
        "xformers_tiled_matmul_source", source_path
    )
    source = importlib.util.module_from_spec(source_spec)
    source_spec.loader.exec_module(source)
    return source


def main():
    # One input tile times three weight tiles: fused Q/K/V projection.
    tokens, hidden, projection = 4096, 4096, 4096
    a = [[torch.randn((tokens, hidden), device="cuda", dtype=torch.float16)]]
    b = [[
        torch.randn((hidden, projection), device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]]
    c = [[
        torch.empty((tokens, projection), device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]]
    source = load_source()

    source._launch_triton_matmul(a, b, c, [tokens], [projection] * 3, [hidden])
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    source._launch_triton_matmul(a, b, c, [tokens], [projection] * 3, [hidden])
    end.record()
    torch.cuda.synchronize()

    means = [tensor.float().mean().item() for tensor in c[0]]
    print(f"input tile: shape={tuple(a[0][0].shape)}, dtype={a[0][0].dtype}")
    print(f"weight tiles=3, each shape={tuple(b[0][0].shape)}")
    print(f"output tiles=3, each shape={tuple(c[0][0].shape)}, means={means}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
