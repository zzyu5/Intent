import importlib.util
from pathlib import Path

import torch


ROOT = next(parent for parent in Path(__file__).parents if parent.name == "vllm")
SPEC = importlib.util.spec_from_file_location("intent_vllm_runtime", ROOT / "support" / "runtime.py")
RUNTIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME)


def main():
    source = RUNTIME.load_source(Path(__file__).with_name("triton_decode_attention.py"), "intent_vllm_paged_gqa")
    batch, query_heads, kv_heads, head_dim = 16, 32, 8, 128
    sequence, page_size, splits = 8192, 16, 8
    pages_per_sequence = (sequence + page_size - 1) // page_size
    pages = batch * pages_per_sequence
    q = torch.randn((batch, query_heads, head_dim), device="cuda", dtype=torch.float16)
    k = torch.randn((pages, page_size, kv_heads, head_dim), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    page_table = torch.arange(pages, device="cuda", dtype=torch.int32).view(batch, pages_per_sequence)
    lengths = torch.full((batch,), sequence, device="cuda", dtype=torch.int32)
    output = torch.empty_like(q)
    lse = torch.empty((batch, query_heads), device="cuda", dtype=torch.float32)
    workspace = torch.empty((batch, query_heads, splits, head_dim + 1), device="cuda", dtype=torch.float32)

    def call():
        source.decode_attention_fwd(
            q, k, v, output, lse, page_table, lengths, workspace, splits,
            head_dim ** -0.5, page_size=page_size,
        )
        return output

    result, latency = RUNTIME.elapsed_ms(call)
    print(f"B={batch} QH={query_heads} KVH={kv_heads} S={sequence} D={head_dim} page={page_size}")
    print(f"output={tuple(result.shape)} dtype={result.dtype} finite={torch.isfinite(result).all().item()}")
    print(f"latency_ms={latency:.3f}")


if __name__ == "__main__":
    main()
