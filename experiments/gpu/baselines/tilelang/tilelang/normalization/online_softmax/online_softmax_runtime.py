import ast
from pathlib import Path

import torch


def load_softmax(rows, columns):
    source_path = Path(__file__).with_name("online_softmax.py")
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    tree.body = [node for node in tree.body if node.end_lineno <= 45]
    namespace = {
        "__file__": str(source_path),
        "__name__": "tilelang_online_softmax_source",
        "M": rows,
        "N": columns,
    }
    exec(compile(tree, str(source_path), "exec"), namespace)
    return namespace["softmax_kernel"]


def main():
    rows, columns = 8192, 8192
    x = torch.randn((rows, columns), device="cuda", dtype=torch.float16)
    softmax = load_softmax(rows, columns)

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
