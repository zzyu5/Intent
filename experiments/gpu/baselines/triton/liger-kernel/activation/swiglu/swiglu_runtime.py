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
swiglu_backward = source.swiglu_backward
STATE = {}


def upstream(arguments):
    if len(arguments) == 2 or (
        len(arguments) == 3
        and arguments[2].shape[-1] == 2 * arguments[0].shape[-1]
    ):
        gate, up = arguments[:2]
        shape = tuple(gate.shape)
        features = shape[-1]
        rows = gate.numel() // features
        key = (shape, gate.dtype, gate.device)
        if key not in STATE:
            STATE[key] = torch.empty_like(gate).reshape(rows, features)
        output = STATE[key]
        gate_rows = gate.reshape(rows, features)
        up_rows = up.reshape(rows, features)
        block, num_warps = source.calculate_settings(features)
        source._swiglu_forward_kernel[(rows,)](
            gate_rows,
            up_rows,
            output,
            output.stride(0),
            n_cols=features,
            BLOCK_SIZE=block,
            num_warps=num_warps,
        )
        return output.view(shape)
    if len(arguments) == 3:
        gradient, gate, up = arguments
        shape = tuple(gradient.shape)
        features = shape[-1]
        rows = gradient.numel() // features
        gradient_rows = gradient.reshape(rows, features)
        gate_rows = gate.reshape(rows, features)
        up_rows = up.reshape(rows, features)
        block, num_warps = source.calculate_settings(features)
        source._swiglu_backward_kernel[(rows,)](
            gradient_rows,
            gate_rows,
            up_rows,
            gradient_rows.stride(0),
            n_cols=features,
            BLOCK_SIZE=block,
            num_warps=num_warps,
        )
        return gate_rows.view(shape), up_rows.view(shape)
    raise RuntimeError("SwiGLU baseline expects forward or backward arguments")


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
