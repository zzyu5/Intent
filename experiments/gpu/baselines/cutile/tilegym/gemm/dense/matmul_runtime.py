import importlib.util
import sys
from pathlib import Path

import tilegym
import torch


SOURCE = Path(__file__).with_name("matmul.py")
spec = importlib.util.spec_from_file_location("tilegym.ops.cutile._local_matmul", SOURCE)
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)


def main():
    m_size, n_size, k_size = 8192, 11008, 4096
    a = torch.randn(m_size, k_size, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k_size, n_size, device="cuda", dtype=torch.bfloat16)

    output = source.matmul(a, b)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = source.matmul(a, b)
    end.record()
    torch.cuda.synchronize()

    print(f"A={tuple(a.shape)} B={tuple(b.shape)} dtype={a.dtype}")
    print(f"output={tuple(output.shape)} mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
