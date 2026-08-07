import importlib.util
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[2] / "support"))

source_path = Path(__file__).with_name("fused_add_rms_norm.py")
spec = importlib.util.spec_from_file_location(
    "liger_kernel.ops.fused_add_rms_norm", source_path
)
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)
fused_add_rms_norm_forward = source.fused_add_rms_norm_forward


def main():
    tokens, hidden = 8192, 4096
    x = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn((hidden,), device="cuda", dtype=torch.bfloat16)

    fused_add_rms_norm_forward(x, residual, weight, 1e-6, 0.0, "llama")
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output, residual_out, _, _, _, _ = fused_add_rms_norm_forward(
        x, residual, weight, 1e-6, 0.0, "llama"
    )
    end.record()
    torch.cuda.synchronize()

    print(f"input/residual: shape={tuple(x.shape)}, dtype={x.dtype}")
    print(f"weight: shape={tuple(weight.shape)}, casting_mode=llama")
    print(f"normalized: shape={tuple(output.shape)}, mean={output.float().mean().item():.6f}")
    print(f"residual_out: shape={tuple(residual_out.shape)}, mean={residual_out.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
