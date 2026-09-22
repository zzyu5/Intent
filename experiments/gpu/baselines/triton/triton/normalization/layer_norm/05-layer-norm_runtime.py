import ast
from pathlib import Path

import torch


def load_layer_norm():
    source_path = Path(__file__).with_name("05-layer-norm.py")
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    tree.body = [node for node in tree.body if node.end_lineno <= 297]
    namespace = {"__file__": str(source_path), "__name__": "triton_layer_norm_source"}
    exec(compile(tree, str(source_path), "exec"), namespace)
    return namespace["layer_norm"]


def main():
    tokens, hidden = 8192, 4096
    x = torch.randn((tokens, hidden), device="cuda", dtype=torch.float16)
    weight = torch.randn((hidden,), device="cuda", dtype=torch.float16)
    bias = torch.randn((hidden,), device="cuda", dtype=torch.float16)
    layer_norm = load_layer_norm()

    layer_norm(x, (hidden,), weight, bias, 1e-5)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = layer_norm(x, (hidden,), weight, bias, 1e-5)
    end.record()
    torch.cuda.synchronize()

    print(f"input: shape={tuple(x.shape)}, dtype={x.dtype}")
    print(f"weight: shape={tuple(weight.shape)}, dtype={weight.dtype}")
    print(f"output: shape={tuple(output.shape)}, dtype={output.dtype}, mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
