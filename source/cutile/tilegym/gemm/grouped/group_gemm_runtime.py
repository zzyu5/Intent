import importlib.util
import sys
from pathlib import Path

import tilegym
import torch


SOURCE = Path(__file__).with_name("group_gemm.py")
spec = importlib.util.spec_from_file_location("tilegym.ops.cutile._local_group_gemm", SOURCE)
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)


def main():
    group_rows = (256, 512, 1024, 2048)
    k_size = n_size = 4096
    group_a = [
        torch.randn(rows, k_size, device="cuda", dtype=torch.bfloat16)
        for rows in group_rows
    ]
    group_b = [
        torch.randn(k_size, n_size, device="cuda", dtype=torch.bfloat16)
        for _ in group_rows
    ]

    outputs = source.group_gemm(group_a, group_b)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    outputs = source.group_gemm(group_a, group_b)
    end.record()
    torch.cuda.synchronize()

    print(f"group_A={[tuple(x.shape) for x in group_a]}")
    print(f"group_B={[tuple(x.shape) for x in group_b]} dtype={group_a[0].dtype}")
    print(
        f"outputs={[tuple(x.shape) for x in outputs]} "
        f"mean={torch.stack([x.float().mean() for x in outputs]).mean().item():.6f}"
    )
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
