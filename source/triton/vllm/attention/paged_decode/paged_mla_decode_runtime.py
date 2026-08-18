import importlib.util
from pathlib import Path

import torch


ROOT = next(parent for parent in Path(__file__).parents if parent.name == "vllm")
SPEC = importlib.util.spec_from_file_location("intent_vllm_runtime", ROOT / "support" / "runtime.py")
RUNTIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME)


def main():
    source = RUNTIME.load_source(Path(__file__).with_name("triton_decode_attention.py"), "intent_vllm_paged_mla")
    batch, query_heads, kv_heads = 8, 128, 1
    sequence, key_dim, latent_dim = 8192, 576, 512
    page_size, splits = 16, 8
    pages_per_sequence = (sequence + page_size - 1) // page_size
    pages = batch * pages_per_sequence
    q = torch.randn((batch, query_heads, key_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((pages, page_size, kv_heads, key_dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((pages, page_size, kv_heads, latent_dim), device="cuda", dtype=torch.bfloat16)
    page_table = torch.arange(pages, device="cuda", dtype=torch.int32).view(batch, pages_per_sequence)
    lengths = torch.full((batch,), sequence, device="cuda", dtype=torch.int32)
    output = torch.empty((batch, query_heads, latent_dim), device="cuda", dtype=torch.bfloat16)
    lse = torch.empty((batch, query_heads), device="cuda", dtype=torch.float32)
    workspace = torch.empty((batch, query_heads, splits, latent_dim + 1), device="cuda", dtype=torch.float32)

    def call():
        source.decode_attention_fwd(
            q, k, v, output, lse, page_table, lengths, workspace, splits,
            key_dim ** -0.5, page_size=page_size, is_mla=True,
        )
        return output

    result, latency = RUNTIME.elapsed_ms(call)
    print(f"B={batch} QH={query_heads} KVH={kv_heads} S={sequence} latent={latent_dim} rope={key_dim-latent_dim}")
    print(f"output={tuple(result.shape)} dtype={result.dtype} finite={torch.isfinite(result).all().item()}")
    print(f"latency_ms={latency:.3f}")


if __name__ == "__main__":
    main()
