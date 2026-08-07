import importlib.util
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[2] / "support"))

source_path = Path(__file__).with_name("swiglu.py")
spec = importlib.util.spec_from_file_location("liger_kernel.ops.swiglu", source_path)
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)
swiglu_forward = source.swiglu_forward


def main():
    tokens, intermediate = 8192, 14336
    gate = torch.randn((tokens, intermediate), device="cuda", dtype=torch.bfloat16)
    up = torch.randn_like(gate)

    swiglu_forward(gate, up)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _, _, output = swiglu_forward(gate, up)
    end.record()
    torch.cuda.synchronize()

    print(f"gate/up: shape={tuple(gate.shape)}, dtype={gate.dtype}")
    print(f"output: shape={tuple(output.shape)}, mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
