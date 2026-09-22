import importlib.util
from pathlib import Path

import torch


SOURCE = Path(__file__).with_name("example_grouped_gemm_fwd.py")
spec = importlib.util.spec_from_file_location("local_tilelang_grouped_gemm", SOURCE)
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def main():
    group_sizes = (256, 512, 1024, 2048)
    k_size = n_size = 4096
    block_m, block_n, block_k = 64, 128, 64
    a, b, sizes, offsets, padded_offsets = source.construct_inputs(
        group_sizes, k_size, n_size, False, block_m, torch.device("cuda"), torch.float16
    )

    output = source.grouped_gemm(
        a, b, sizes, offsets, padded_offsets, group_sizes,
        block_m, block_n, block_k, False, 2, 256,
    )
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = source.grouped_gemm(
        a, b, sizes, offsets, padded_offsets, group_sizes,
        block_m, block_n, block_k, False, 2, 256,
    )
    end.record()
    torch.cuda.synchronize()

    print(f"groups={group_sizes} A={tuple(a.shape)} B={tuple(b.shape)} dtype={a.dtype}")
    print(f"output={tuple(output.shape)} mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
