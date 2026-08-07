import importlib.util
from pathlib import Path

import torch


SOURCE = Path(__file__).with_name("cross_entropy.py")
spec = importlib.util.spec_from_file_location("local_flash_attn_cross_entropy", SOURCE)
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def main():
    tokens, vocab_size = 8192, 32768
    logits = torch.randn(tokens, vocab_size, device="cuda", dtype=torch.bfloat16)
    labels = torch.randint(0, vocab_size, (tokens,), device="cuda", dtype=torch.long)

    losses, z_losses = source.cross_entropy_loss(logits, labels, lse_square_scale=1.0e-4)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    losses, z_losses = source.cross_entropy_loss(logits, labels, lse_square_scale=1.0e-4)
    end.record()
    torch.cuda.synchronize()

    print(f"logits={tuple(logits.shape)} labels={tuple(labels.shape)} dtype={logits.dtype}")
    print(f"loss={losses.mean().item():.6f} z_loss={z_losses.mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
