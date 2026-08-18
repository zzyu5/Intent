import importlib.util
from pathlib import Path

import torch


def main():
    root = next(parent for parent in Path(__file__).parents if parent.name == "meta-applied-ai")
    spec = importlib.util.spec_from_file_location("intent_meta_runtime", root / "support" / "runtime.py")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    source = runtime.load_source(Path(__file__).with_name("flash_backward.py"), "intent_flash_forward")
    batch, heads, sequence, head_dim = 2, 16, 2048, 128
    shape = (batch, heads, sequence, head_dim)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    output = torch.empty_like(q)
    lse = torch.empty((batch, heads, sequence), device="cuda", dtype=torch.float32)
    result, latency = runtime.elapsed_ms(lambda: source.flash(q, k, v, output, lse))
    print(f"Q/K/V={shape} dtype={q.dtype} causal=True")
    print(f"output={tuple(result.shape)} dtype={result.dtype} mean={result.float().mean().item():.6f}")
    print(f"latency_ms={latency:.3f}")


if __name__ == "__main__":
    main()
