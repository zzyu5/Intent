import ast
from pathlib import Path

import torch


def load_matmul():
    source_path = Path(__file__).with_name("03-matrix-multiplication.py")
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    tree.body = [node for node in tree.body if node.end_lineno <= 352]
    namespace = {"__file__": str(source_path), "__name__": "triton_matmul_source"}
    exec(compile(tree, str(source_path), "exec"), namespace)
    return namespace["matmul"]


def main():
    # Mixtral 8x7B MLP up projection, evaluated on 4096 tokens.
    m, k, n = 4096, 4096, 14336
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    matmul = load_matmul()

    matmul(a, b)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = matmul(a, b)
    end.record()
    torch.cuda.synchronize()

    print(f"A: shape={tuple(a.shape)}, dtype={a.dtype}")
    print(f"B: shape={tuple(b.shape)}, dtype={b.dtype}")
    print(f"output: shape={tuple(output.shape)}, dtype={output.dtype}, mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
