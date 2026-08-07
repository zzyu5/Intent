import importlib.util
from pathlib import Path

import torch


def load_source():
    source_path = Path(__file__).with_name("k_scaled_index_add.py")
    spec = importlib.util.spec_from_file_location(
        "xformers_scaled_index_add_source", source_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    destination_rows, routed_rows, hidden = 65536, 32768, 4096
    destination = torch.randn(
        (destination_rows, 1, hidden), device="cuda", dtype=torch.float16
    )
    routed = torch.randn(
        (routed_rows, 1, hidden), device="cuda", dtype=torch.float16
    )
    indices = torch.arange(
        0, routed_rows * 2, 2, device="cuda", dtype=torch.int64
    )
    scaling = torch.randn((hidden,), device="cuda", dtype=torch.float16)
    source = load_source()

    source.scaled_index_add_fwd(destination, indices, routed, scaling, 1.0)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    source.scaled_index_add_fwd(destination, indices, routed, scaling, 1.0)
    end.record()
    torch.cuda.synchronize()

    print(f"destination: shape={tuple(destination.shape)}, dtype={destination.dtype}")
    print(f"routed source: shape={tuple(routed.shape)}, unique destinations=True")
    print(f"feature scaling: shape={tuple(scaling.shape)}")
    print(f"updated destination mean={destination.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
