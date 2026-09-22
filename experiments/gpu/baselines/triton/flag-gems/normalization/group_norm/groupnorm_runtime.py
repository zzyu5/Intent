import importlib.util
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")

import torch
import triton
import triton.language as tl


SOURCE = Path(__file__).with_name("groupnorm.py")
SOURCE_MODULE = None
STATE = {}


def _load_source():
    global SOURCE_MODULE
    if SOURCE_MODULE is not None:
        return SOURCE_MODULE
    runtime = types.ModuleType("flag_gems.runtime")
    runtime.torch_device_fn = types.SimpleNamespace(device=torch.cuda.device)

    utils = types.ModuleType("flag_gems.utils")
    utils.libentry = lambda: (lambda function: function)
    utils.tl_extra_shim = types.SimpleNamespace(rsqrt=tl.rsqrt)
    utils.triton_lang_extension = types.SimpleNamespace(program_id=tl.program_id)

    package = types.ModuleType("flag_gems")
    package.__path__ = []
    sys.modules["flag_gems"] = package
    sys.modules["flag_gems.runtime"] = runtime
    sys.modules["flag_gems.utils"] = utils

    spec = importlib.util.spec_from_file_location("flag_gems_groupnorm_source", SOURCE)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    SOURCE_MODULE = source
    return source


def upstream(arguments):
    x, grad_y, weight, mean, rstd = arguments
    batch, channels, spatial = x.shape
    groups = mean.shape[1]
    group_size = channels // groups
    key = (tuple(x.shape), groups, x.dtype, x.device)
    if key not in STATE:
        STATE[key] = (
            torch.empty_like(x),
            torch.empty_like(weight),
            torch.empty_like(weight),
        )
    grad_x, grad_weight, grad_bias = STATE[key]
    source = _load_source()
    source.group_norm_backward_kernel[(batch * groups,)](
        grad_y,
        x,
        weight,
        mean,
        rstd,
        groups,
        group_size,
        grad_x,
        channels,
        spatial,
        BLOCK_GROUP_SIZE=triton.next_power_of_2(group_size),
    )
    source.weight_bias_backward_kernel[(channels, 1, 1)](
        grad_y,
        x,
        mean,
        rstd,
        grad_weight,
        grad_bias,
        groups,
        group_size,
        batch,
        channels,
        spatial,
        BLOCK_N=triton.next_power_of_2(batch),
        BLOCK_HW=triton.next_power_of_2(spatial),
    )
    return grad_x, grad_weight, grad_bias


def main():
    batch, channels, height, width, groups = 32, 256, 32, 32, 32
    spatial = height * width
    x = torch.randn(
        batch, channels, height, width, device="cuda", dtype=torch.float16
    )
    weight = torch.randn(channels, device="cuda", dtype=torch.float16)
    bias = torch.randn(channels, device="cuda", dtype=torch.float16)
    grad_y = torch.randn_like(x)
    source = _load_source()

    y, mean, rstd = source.group_norm(
        x, weight, bias, batch, channels, spatial, groups
    )
    grad_x, grad_weight, grad_bias = source.group_norm_backward(
        grad_y,
        x,
        mean,
        rstd,
        weight,
        batch,
        channels,
        spatial,
        groups,
        (True, True, True),
    )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    source.group_norm_backward(
        grad_y,
        x,
        mean,
        rstd,
        weight,
        batch,
        channels,
        spatial,
        groups,
        (True, True, True),
    )
    end.record()
    torch.cuda.synchronize()

    print(
        f"x={tuple(x.shape)} groups={groups} dtype={x.dtype} "
        f"y={tuple(y.shape)}"
    )
    print(
        f"grad_x={tuple(grad_x.shape)} grad_weight={tuple(grad_weight.shape)} "
        f"grad_bias={tuple(grad_bias.shape)}"
    )
    print(f"backward_latency_ms={start.elapsed_time(end):.4f}")


if __name__ == "__main__":
    main()
