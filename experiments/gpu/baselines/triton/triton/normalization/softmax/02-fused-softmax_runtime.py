import ast
from pathlib import Path

import torch


def load_softmax():
    source_path = Path(__file__).with_name("02-fused-softmax.py")
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    tree.body = [node for node in tree.body if node.end_lineno <= 175]
    namespace = {"__file__": str(source_path), "__name__": "triton_fused_softmax_source"}
    exec(compile(tree, str(source_path), "exec"), namespace)
    return namespace["softmax"]


def main():
    rows = 8192
    columns = 8192
    x = torch.randn((rows, columns), device="cuda", dtype=torch.float16)
    softmax = load_softmax()

    softmax(x)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = softmax(x)
    end.record()
    torch.cuda.synchronize()

    print(f"input: shape={tuple(x.shape)}, dtype={x.dtype}")
    print(f"output: shape={tuple(output.shape)}, dtype={output.dtype}, mean={output.float().mean().item():.8f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
