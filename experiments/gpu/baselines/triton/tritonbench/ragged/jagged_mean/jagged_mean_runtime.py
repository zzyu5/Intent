import importlib.util
from pathlib import Path

import torch
import triton


SOURCE = Path(__file__).with_name("kernels.py")
SOURCE_MODULE = None
STATE = {}


def _load_source():
    global SOURCE_MODULE
    if SOURCE_MODULE is not None:
        return SOURCE_MODULE
    spec = importlib.util.spec_from_file_location("tritonbench_jagged_mean", SOURCE)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    SOURCE_MODULE = source
    return source


def upstream(arguments):
    values, offsets = arguments
    batch = offsets.shape[0] - 1
    features = values.shape[1]
    key = (batch, features, values.dtype, values.device)
    if key not in STATE:
        STATE[key] = torch.empty(
            batch, features, device=values.device, dtype=values.dtype
        )
    output = STATE[key]
    kernel = _load_source().triton_jagged_mean_kernel_simple_fused_sum_then_buffer
    grid = lambda meta: (
        batch * triton.cdiv(features, meta["BLOCK_SIZE_M"]),
    )
    kernel[grid](
        values,
        offsets,
        output,
        M=features,
        MAX_SEQLEN=128,
    )
    return output


def main():
    batch, features, max_length = 512, 128, 128
    lengths = (
        torch.arange(batch, device="cuda", dtype=torch.int32) % max_length
    ) + 1
    offsets = torch.empty(batch + 1, device="cuda", dtype=torch.int32)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(lengths, dim=0)
    values = torch.randn(
        int(offsets[-1].item()), features, device="cuda", dtype=torch.float32
    )
    output = upstream((values, offsets))
    torch.cuda.synchronize()
    reference = torch.segment_reduce(values, "mean", lengths=lengths)
    max_error = (output - reference).abs().max().item()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    upstream((values, offsets))
    end.record()
    torch.cuda.synchronize()

    print(
        f"values={tuple(values.shape)} offsets={tuple(offsets.shape)} "
        f"output={tuple(output.shape)}"
    )
    print(f"max_error={max_error:.6g} latency_ms={start.elapsed_time(end):.4f}")


if __name__ == "__main__":
    main()
