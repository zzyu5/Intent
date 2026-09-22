import importlib.util
import sys
import types
from pathlib import Path

import torch


def load_source():
    package_root = Path(__file__).parents[2]
    xformers_package = types.ModuleType("xformers")
    xformers_package.__path__ = []
    triton_package = types.ModuleType("xformers.triton")
    triton_package.__path__ = []
    sys.modules["xformers"] = xformers_package
    sys.modules["xformers.triton"] = triton_package

    importing_path = package_root / "support" / "triton" / "importing.py"
    importing_spec = importlib.util.spec_from_file_location(
        "xformers.triton.importing", importing_path
    )
    importing_module = importlib.util.module_from_spec(importing_spec)
    sys.modules[importing_spec.name] = importing_module
    importing_spec.loader.exec_module(importing_module)

    source_path = Path(__file__).with_name("rmsnorm_kernels.py")
    source_spec = importlib.util.spec_from_file_location(
        "xformers_rmsnorm_source", source_path
    )
    source = importlib.util.module_from_spec(source_spec)
    source_spec.loader.exec_module(source)
    return source


def main():
    tokens, hidden = 8192, 4096
    x = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn((hidden,), device="cuda", dtype=torch.bfloat16)
    source = load_source()

    source._rms_norm_forward(x, weight, 1e-6)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = source._rms_norm_forward(x, weight, 1e-6)
    end.record()
    torch.cuda.synchronize()
    rms_ms = start.elapsed_time(end)

    residual_input = x.clone()
    source._rms_norm_add_forward(residual_input, residual, weight, 1e-6)
    torch.cuda.synchronize()
    start.record()
    residual_output = source._rms_norm_add_forward(
        residual_input, residual, weight, 1e-6
    )
    end.record()
    torch.cuda.synchronize()

    print(f"input/residual: shape={tuple(x.shape)}, dtype={x.dtype}")
    print(f"RMSNorm output: shape={tuple(output.shape)}, mean={output.float().mean().item():.6f}")
    print(f"RMSNorm latency_ms={rms_ms:.3f}")
    print(
        f"residual RMSNorm output: shape={tuple(residual_output.shape)}, "
        f"mean={residual_output.float().mean().item():.6f}"
    )
    print(f"residual_RMSNorm_latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
