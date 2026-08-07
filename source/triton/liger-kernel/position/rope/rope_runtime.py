import importlib.util
from pathlib import Path

import torch


def load_rope_forward():
    source_path = Path(__file__).with_name("rope.py")
    spec = importlib.util.spec_from_file_location("liger_rope_source", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.rope_forward


def main():
    batch, q_heads, kv_heads, sequence, head_dim = 4, 32, 8, 4096, 128
    rotary_base = 10000.0
    q = torch.randn((batch, q_heads, sequence, head_dim), device="cuda", dtype=torch.float16)
    k = torch.randn((batch, kv_heads, sequence, head_dim), device="cuda", dtype=torch.float16)
    positions = torch.arange(sequence, device="cuda", dtype=torch.float32)
    inv_freq = 1.0 / (rotary_base ** (torch.arange(0, head_dim, 2, device="cuda", dtype=torch.float32) / head_dim))
    frequencies = torch.outer(positions, inv_freq)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cos = embedding.cos().unsqueeze(0).to(torch.float16)
    sin = embedding.sin().unsqueeze(0).to(torch.float16)
    rope_forward = load_rope_forward()

    rope_forward(q, k, cos, sin)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    q_out, k_out, _, _ = rope_forward(q, k, cos, sin)
    end.record()
    torch.cuda.synchronize()

    print(f"Q: shape={tuple(q.shape)}, K: shape={tuple(k.shape)}, dtype={q.dtype}")
    print(f"cos/sin: shape={tuple(cos.shape)}, rotary_base={rotary_base}")
    print(f"Q output: shape={tuple(q_out.shape)}, mean={q_out.float().mean().item():.6f}")
    print(f"K output: shape={tuple(k_out.shape)}, mean={k_out.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
