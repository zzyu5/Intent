import importlib.util
from pathlib import Path

import torch


def load_source():
    source_path = Path(__file__).with_name("k_index_select_cat.py")
    spec = importlib.util.spec_from_file_location(
        "xformers_index_select_source", source_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    source_rows, selected_rows, hidden = 65536, 32768, 4096
    source_tensor = torch.randn(
        (source_rows, hidden), device="cuda", dtype=torch.float16
    )
    indices = torch.arange(
        0, selected_rows * 2, 2, device="cuda", dtype=torch.int64
    )
    output = torch.empty(
        (selected_rows, hidden), device="cuda", dtype=torch.float16
    )
    source = load_source()

    source.index_select_cat_fwd(output, source_tensor, indices)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    source.index_select_cat_fwd(output, source_tensor, indices)
    end.record()
    torch.cuda.synchronize()

    print(f"source: shape={tuple(source_tensor.shape)}, dtype={source_tensor.dtype}")
    print(f"indices: shape={tuple(indices.shape)}, unique=True")
    print(f"output: shape={tuple(output.shape)}, mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
