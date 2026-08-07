import importlib.util
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[2] / "support"))

source_path = Path(__file__).with_name("embedding.py")
spec = importlib.util.spec_from_file_location(
    "liger_kernel.ops.experimental.embedding", source_path
)
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)
LigerEmbeddingFunction = source.LigerEmbeddingFunction


def main():
    batch, sequence, vocabulary, hidden = 8, 2048, 32768, 4096
    table = torch.randn(
        (vocabulary, hidden), device="cuda", dtype=torch.bfloat16
    )
    indices = torch.randint(
        0, vocabulary, (batch, sequence), device="cuda", dtype=torch.int64
    )

    LigerEmbeddingFunction.apply(table, indices)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = LigerEmbeddingFunction.apply(table, indices)
    end.record()
    torch.cuda.synchronize()

    print(f"table: shape={tuple(table.shape)}, dtype={table.dtype}")
    print(f"indices: shape={tuple(indices.shape)}, vocabulary={vocabulary}")
    print(f"output: shape={tuple(output.shape)}, mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
