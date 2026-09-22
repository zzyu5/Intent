import importlib.util
import sys
from pathlib import Path

import tilegym
import torch


SOURCE = Path(__file__).with_name("softmax.py")
UTILS = Path(__file__).parents[2] / "support" / "utils.py"

utils_spec = importlib.util.spec_from_file_location("tilegym.ops.cutile.utils", UTILS)
utils = importlib.util.module_from_spec(utils_spec)
sys.modules[utils_spec.name] = utils
utils_spec.loader.exec_module(utils)
spec = importlib.util.spec_from_file_location("tilegym.ops.cutile._local_softmax", SOURCE)
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)


def main():
    rows, columns = 8192, 32768
    x = torch.randn(rows, columns, device="cuda", dtype=torch.bfloat16)

    output = source.softmax(x, use_chunked=True)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = source.softmax(x, use_chunked=True)
    end.record()
    torch.cuda.synchronize()

    print(f"x={tuple(x.shape)} dtype={x.dtype} algorithm=chunked")
    print(
        f"output={tuple(output.shape)} row_sum_mean={output.float().sum(dim=-1).mean().item():.6f}"
    )
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
