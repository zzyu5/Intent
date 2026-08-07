import importlib.util
import sys
from pathlib import Path

import torch


SOURCE = Path(__file__).with_name("example_gemm_sp.py")
sys.path.insert(0, str(SOURCE.parent))
spec = importlib.util.spec_from_file_location("local_tilelang_sparse_gemm", SOURCE)
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
import sparse_utils


def main():
    m_size, n_size, k_size = 8192, 14336, 8192
    dense_a = source.randn_semi_sparse(
        m_size, k_size, device="cuda", dtype=torch.float16
    )
    b = torch.randn(k_size, n_size, device="cuda", dtype=torch.float16)
    sparse_a, metadata = sparse_utils.torch_compress(dense_a, meta_dtype=torch.int16)
    kernel = source.matmul_sp_fp16(
        m_size,
        n_size,
        k_size,
        source.T.float,
        source.T.int16,
        128,
        128,
        64,
        2,
        128,
        source.T.GemmWarpPolicy.Square,
        True,
    )

    output = kernel(sparse_a, metadata, b)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = kernel(sparse_a, metadata, b)
    end.record()
    torch.cuda.synchronize()

    print(
        f"dense_A={tuple(dense_a.shape)} sparse_A={tuple(sparse_a.shape)} "
        f"metadata={tuple(metadata.shape)} B={tuple(b.shape)}"
    )
    print(f"output={tuple(output.shape)} mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
