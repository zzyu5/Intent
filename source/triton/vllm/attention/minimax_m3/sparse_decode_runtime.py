import importlib.util
from pathlib import Path

import torch


ROOT = next(parent for parent in Path(__file__).parents if parent.name == "vllm")
SPEC = importlib.util.spec_from_file_location("intent_vllm_runtime", ROOT / "support" / "runtime.py")
RUNTIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME)


def main():
    source = RUNTIME.load_source(Path(__file__).with_name("sparse_attn.py"), "intent_vllm_minimax_sparse")
    batch, query_heads, kv_heads, head_dim = 8, 32, 8, 128
    sequence, selected_blocks = 8192, 32
    block_size = source.SPARSE_BLOCK_SIZE
    blocks_per_sequence = (sequence + block_size - 1) // block_size
    blocks = batch * blocks_per_sequence
    q = torch.randn((batch, query_heads, head_dim), device="cuda", dtype=torch.float16)
    kv = torch.randn((blocks, kv_heads, block_size, 2 * head_dim), device="cuda", dtype=torch.float16)
    block_table = torch.arange(blocks, device="cuda", dtype=torch.int32).view(batch, blocks_per_sequence)
    topk = torch.arange(selected_blocks, device="cuda", dtype=torch.int32)
    topk = topk.view(1, 1, selected_blocks).expand(kv_heads, batch, selected_blocks).contiguous()
    lengths = torch.full((batch,), sequence, device="cuda", dtype=torch.int32)
    output = torch.empty_like(q)

    def call():
        source.minimax_m3_sparse_attn_decode(
            q, kv, topk, block_table, lengths, kv_heads,
            head_dim ** -0.5, output, decode_query_len=1,
        )
        return output

    result, latency = RUNTIME.elapsed_ms(call)
    print(f"B={batch} QH={query_heads} KVH={kv_heads} S={sequence} D={head_dim} topk_blocks={selected_blocks}")
    print(f"output={tuple(result.shape)} dtype={result.dtype} finite={torch.isfinite(result).all().item()}")
    print(f"latency_ms={latency:.3f}")


if __name__ == "__main__":
    main()
