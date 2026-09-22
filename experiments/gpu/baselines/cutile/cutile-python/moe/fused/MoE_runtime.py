import importlib.util
from pathlib import Path

import torch


def load_source():
    source_path = Path(__file__).with_name("MoE.py")
    spec = importlib.util.spec_from_file_location("cutile_moe_source", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    # DeepSeek-V3 hidden/expert widths with a memory-feasible eight-expert shard.
    tokens, hidden, experts, intermediate, top_k = 8192, 7168, 8, 2048, 4
    hidden_states = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(
        (experts, intermediate * 2, hidden), device="cuda", dtype=torch.bfloat16
    )
    w2 = torch.randn((experts, hidden, intermediate), device="cuda", dtype=torch.bfloat16)
    token_ids = torch.arange(tokens, device="cuda")[:, None]
    route_offsets = torch.arange(top_k, device="cuda")[None, :]
    topk_ids = ((token_ids + route_offsets) % experts).to(torch.int64)
    topk_weights = torch.softmax(
        torch.randn((tokens, top_k), device="cuda", dtype=torch.float32), dim=-1
    ).to(torch.bfloat16)
    source = load_source()

    source.cutile_moe(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        tile_m=128,
        tile_n=128,
        tile_k=64,
    )
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = source.cutile_moe(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        tile_m=128,
        tile_n=128,
        tile_k=64,
    )
    end.record()
    torch.cuda.synchronize()

    print(f"input: shape={tuple(hidden_states.shape)}, dtype={hidden_states.dtype}")
    print(f"experts={experts}, top_k={top_k}, expert_dim={intermediate}")
    print(f"W1={tuple(w1.shape)}, W2={tuple(w2.shape)}")
    print(f"output: shape={tuple(output.shape)}, dtype={output.dtype}, mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
