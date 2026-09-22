import importlib.util
import sys
import types
from pathlib import Path

import torch


SOURCE = Path(__file__).with_name("layer_norm.py")
FLASH_ATTN_ROOT = SOURCE.parents[2]


def load_package(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load_package("flash_attn", FLASH_ATTN_ROOT)
load_package("flash_attn.utils", SOURCE.parent / "support")
load_package("flash_attn.ops", SOURCE.parent)
load_package("flash_attn.ops.triton", SOURCE.parent)
load_module("flash_attn.utils.torch", SOURCE.parent / "support" / "torch.py")
load_module("flash_attn.utils.library", SOURCE.parent / "support" / "library.py")
source = load_module("flash_attn.ops.triton.layer_norm", SOURCE)
STATE = {}


def upstream(arguments):
    x, weight, bias, _, epsilon = arguments
    rows, features = x.shape
    key = (tuple(x.shape), x.dtype, x.device, weight.dtype, bias.dtype)
    if key not in STATE:
        STATE[key] = (
            torch.empty_like(x),
            torch.empty((rows,), device=x.device, dtype=torch.float32),
            torch.empty((rows,), device=x.device, dtype=torch.float32),
        )
    output, mean, rstd = STATE[key]
    block = min(65536 // x.element_size(), source.triton.next_power_of_2(features))
    if features > block:
        raise NotImplementedError(
            "FlashAttention LayerNorm baseline requires a single fused feature tile"
        )
    source._layer_norm_fwd_1pass_kernel[(rows,)](
        x,
        output,
        weight,
        bias,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        mean,
        rstd,
        x.stride(0),
        output.stride(0),
        0,
        0,
        0,
        0,
        rows,
        features,
        epsilon,
        0.0,
        0,
        False,
        block,
        False,
        False,
        True,
        False,
        False,
        False,
        HAS_X1=False,
        HAS_W1=False,
        HAS_B1=False,
    )
    return output


def main():
    tokens, hidden_size = 8192, 4096
    x = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.ones(hidden_size, device="cuda", dtype=torch.bfloat16)

    output, residual_out = source.rms_norm_fn(
        x, weight, None, residual=residual, prenorm=True
    )
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output, residual_out = source.rms_norm_fn(
        x, weight, None, residual=residual, prenorm=True
    )
    end.record()
    torch.cuda.synchronize()

    print(f"x={tuple(x.shape)} residual={tuple(residual.shape)} dtype={x.dtype}")
    print(
        f"output={tuple(output.shape)} residual_out={tuple(residual_out.shape)} "
        f"mean={output.float().mean().item():.6f}"
    )
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
