import importlib.util
from pathlib import Path

import torch


def load_source():
    source_path = Path(__file__).with_name("LayerNorm.py")
    spec = importlib.util.spec_from_file_location("cutile_layer_norm_source", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    tokens, hidden = 8192, 4096
    x = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((hidden,), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((hidden,), device="cuda", dtype=torch.bfloat16)
    source = load_source()

    source.cutile_layer_norm(x, weight, bias, 1e-5)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = source.cutile_layer_norm(x, weight, bias, 1e-5)
    end.record()
    torch.cuda.synchronize()

    print(f"input: shape={tuple(x.shape)}, dtype={x.dtype}")
    print(f"weight/bias: shape={tuple(weight.shape)}, dtype={weight.dtype}")
    print(f"output: shape={tuple(output.shape)}, dtype={output.dtype}, mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
