import ast
import math
from pathlib import Path

import torch


def load_attention():
    source_path = Path(__file__).with_name("06-fused-attention.py")
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    body = []
    for node in tree.body:
        if node.end_lineno > 621:
            continue
        if isinstance(node, ast.Import) and any(alias.name == "pytest" for alias in node.names):
            continue
        body.append(node)
    tree.body = body
    namespace = {"__file__": str(source_path), "__name__": "triton_fused_attention_source"}
    exec(compile(tree, str(source_path), "exec"), namespace)
    return namespace["attention"]


def main():
    batch, heads, sequence, head_dim = 4, 32, 4096, 128
    shape = (batch, heads, sequence, head_dim)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn(shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, device="cuda", dtype=torch.float16)
    attention = load_attention()
    scale = 1.0 / math.sqrt(head_dim)

    attention(q, k, v, True, scale)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = attention(q, k, v, True, scale)
    end.record()
    torch.cuda.synchronize()

    print(f"Q/K/V: shape={shape}, dtype={q.dtype}, causal=True")
    print(f"output: shape={tuple(output.shape)}, dtype={output.dtype}, mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
