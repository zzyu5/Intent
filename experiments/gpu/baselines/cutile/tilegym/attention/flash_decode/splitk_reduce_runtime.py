import importlib.util
import sys
from pathlib import Path

import tilegym
import torch


SOURCE = Path(__file__).with_name("splitk_reduce.py")
UTILS = Path(__file__).parents[2] / "support" / "utils.py"
utils_spec = importlib.util.spec_from_file_location("tilegym.ops.cutile.utils", UTILS)
utils = importlib.util.module_from_spec(utils_spec)
sys.modules[utils_spec.name] = utils
utils_spec.loader.exec_module(utils)
spec = importlib.util.spec_from_file_location("tilegym.ops.cutile._local_splitk_reduce", SOURCE)
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)


def main():
    batch, heads, splits, head_dim, kv_length = 8, 32, 16, 128, 8192
    partial = torch.randn(
        batch, heads, splits, head_dim, device="cuda", dtype=torch.bfloat16
    )
    partial_lse = torch.randn(batch, heads, splits, device="cuda", dtype=torch.float32)
    output = torch.empty(batch, heads, head_dim, device="cuda", dtype=torch.bfloat16)

    source.splitk_reduce(partial, partial_lse, output, kv_length)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    source.splitk_reduce(partial, partial_lse, output, kv_length)
    end.record()
    torch.cuda.synchronize()

    print(
        f"partial={tuple(partial.shape)} partial_lse={tuple(partial_lse.shape)} "
        f"kv_length={kv_length} dtype={partial.dtype}"
    )
    print(f"output={tuple(output.shape)} mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
