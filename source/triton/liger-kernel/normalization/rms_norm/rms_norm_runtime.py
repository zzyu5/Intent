import importlib.util
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[2] / "support"))

source_path = Path(__file__).with_name("rms_norm.py")
spec = importlib.util.spec_from_file_location("liger_kernel.ops.rms_norm", source_path)
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)
rms_norm_forward = source.rms_norm_forward
STATE = {}


def upstream(arguments):
    x, weight, _, epsilon = arguments
    shape = tuple(x.shape)
    features = shape[-1]
    rows = x.numel() // features
    key = (shape, x.dtype, x.device, weight.dtype)
    if key not in STATE:
        STATE[key] = (
            torch.empty_like(x).reshape(rows, features),
            torch.empty((rows,), device=x.device, dtype=x.dtype),
        )
    output, rstd = STATE[key]
    x_rows = x.reshape(rows, features)
    block, num_warps = source.calculate_settings(features)
    source._rms_norm_forward_kernel[(rows,)](
        output,
        output.stride(0),
        x_rows,
        x_rows.stride(0),
        weight,
        weight.stride(0),
        rstd,
        rstd.stride(0),
        features,
        epsilon,
        0.0,
        source._str_to_casting_mode["none"],
        elementwise_affine=True,
        BLOCK_SIZE=block,
        num_warps=num_warps,
    )
    return output.view(shape)


def main():
    tokens, hidden = 8192, 4096
    x = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((hidden,), device="cuda", dtype=torch.bfloat16)

    rms_norm_forward(x, weight, 1e-6, 0.0, "llama", None)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output, _, _, _, _, _ = rms_norm_forward(
        x, weight, 1e-6, 0.0, "llama", None
    )
    end.record()
    torch.cuda.synchronize()

    print(f"input: shape={tuple(x.shape)}, dtype={x.dtype}")
    print(f"weight: shape={tuple(weight.shape)}, casting_mode=llama")
    print(f"output: shape={tuple(output.shape)}, mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
