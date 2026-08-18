import importlib.util
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")

import torch
import torch.nn.functional as F
import triton


SOURCE = Path(__file__).with_name("max_pool2d_with_indices.py")
SOURCE_MODULE = None
STATE = {}


@triton.jit
def _dtype_minimum(dtype: triton.language.constexpr):
    return -float("inf")


def _load_source():
    global SOURCE_MODULE
    if SOURCE_MODULE is not None:
        return SOURCE_MODULE
    utils = types.ModuleType("flag_gems.utils")
    utils.libentry = lambda: (lambda function: function)
    limits = types.ModuleType("flag_gems.utils.limits")
    limits.get_dtype_min = _dtype_minimum
    package = types.ModuleType("flag_gems")
    package.__path__ = []

    sys.modules["flag_gems"] = package
    sys.modules["flag_gems.utils"] = utils
    sys.modules["flag_gems.utils.limits"] = limits

    spec = importlib.util.spec_from_file_location("flag_gems_max_pool2d", SOURCE)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    SOURCE_MODULE = source
    return source


def upstream(arguments):
    (x,) = arguments
    batch, channels, height, width = x.shape
    output_height = (height + 2 - 3) // 2 + 1
    output_width = (width + 2 - 3) // 2 + 1
    key = (tuple(x.shape), x.dtype, x.device)
    if key not in STATE:
        STATE[key] = (
            torch.empty(
                batch,
                channels,
                output_height,
                output_width,
                device=x.device,
                dtype=x.dtype,
            ),
            torch.empty(
                batch,
                channels,
                output_height,
                output_width,
                device=x.device,
                dtype=torch.int64,
            ),
        )
    output, indices = STATE[key]
    source = _load_source()
    grid = lambda meta: (
        batch * channels,
        triton.cdiv(output_height, meta["BLOCK_H"])
        * triton.cdiv(output_width, meta["BLOCK_W"]),
    )
    source.max_pool2d_forward_kernel[grid](
        x,
        output,
        indices,
        *x.stride(),
        channels,
        height,
        width,
        output_height,
        output_width,
        3,
        3,
        2,
        2,
        1,
        1,
        1,
        1,
    )
    return output, indices


def main():
    batch, channels, height, width = 8, 64, 128, 128
    x = torch.randn(
        batch, channels, height, width, device="cuda", dtype=torch.float16
    )
    values, indices = upstream((x,))
    torch.cuda.synchronize()
    reference_values, reference_indices = F.max_pool2d(
        x, kernel_size=3, stride=2, padding=1, return_indices=True
    )
    max_error = (values - reference_values).abs().max().item()
    indices_equal = torch.equal(indices, reference_indices)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    upstream((x,))
    end.record()
    torch.cuda.synchronize()

    print(
        f"x={tuple(x.shape)} values={tuple(values.shape)} "
        f"indices={tuple(indices.shape)}"
    )
    print(
        f"max_error={max_error:.6g} indices_equal={indices_equal} "
        f"latency_ms={start.elapsed_time(end):.4f}"
    )


if __name__ == "__main__":
    main()
