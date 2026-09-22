import ast
from pathlib import Path

import torch


def load_source():
    source_path = Path(__file__).with_name("AttentionFMHA.py")
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and node.module == "utils.benchmark")
    ]
    namespace = {"__file__": str(source_path), "__name__": "cutile_fmha_source"}
    exec(compile(tree, str(source_path), "exec"), namespace)
    return namespace


def main():
    batch, query_heads, kv_heads, sequence, head_dim = 4, 32, 8, 4096, 128
    q = torch.randn((batch, query_heads, sequence, head_dim), device="cuda", dtype=torch.float16)
    k = torch.randn((batch, kv_heads, sequence, head_dim), device="cuda", dtype=torch.float16)
    v = torch.randn((batch, kv_heads, sequence, head_dim), device="cuda", dtype=torch.float16)
    source = load_source()
    fmha = source["cutile_fmha"]

    fmha(q, k, v, query_group_size=query_heads // kv_heads, causal=True)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = fmha(q, k, v, query_group_size=query_heads // kv_heads, causal=True)
    end.record()
    torch.cuda.synchronize()

    print(f"Q: shape={tuple(q.shape)}, dtype={q.dtype}")
    print(f"K/V: shape={tuple(k.shape)}, query_group_size={query_heads // kv_heads}, causal=True")
    print(f"output: shape={tuple(output.shape)}, dtype={output.dtype}, mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
