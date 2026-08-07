import ast
from pathlib import Path

import torch


def load_source():
    source_path = Path(__file__).with_name("example_gqa_decode.py")
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and node.module == "einops")
    ]
    namespace = {"__file__": str(source_path), "__name__": "tilelang_gqa_decode_source"}
    exec(compile(tree, str(source_path), "exec"), namespace)
    return namespace


def main():
    batch, query_heads, kv_heads, kv_length, head_dim = 32, 32, 8, 8192, 128
    q = torch.randn((batch, query_heads, head_dim), device="cuda", dtype=torch.float16)
    k = torch.randn((batch, kv_length, kv_heads, head_dim), device="cuda", dtype=torch.float16)
    v = torch.randn((batch, kv_length, kv_heads, head_dim), device="cuda", dtype=torch.float16)
    mask = torch.ones((batch, kv_length, kv_heads), device="cuda", dtype=torch.uint8)
    source = load_source()
    config, _ = source["get_heuristic_config"]()
    config["block_N"] = 64
    kernel = source["flashattn"](batch, query_heads, kv_heads, kv_length, head_dim, **config)

    kernel(q, k, v, mask)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = kernel(q, k, v, mask)
    end.record()
    torch.cuda.synchronize()

    print(f"Q: shape={tuple(q.shape)}, dtype={q.dtype}")
    print(f"K/V: shape={tuple(k.shape)}, KV length={kv_length}")
    print(f"output: shape={tuple(output.shape)}, dtype={output.dtype}, mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
