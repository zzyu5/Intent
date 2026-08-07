import importlib.util
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[2] / "support"))

source_path = Path(__file__).with_name("cross_entropy.py")
spec = importlib.util.spec_from_file_location("liger_kernel.ops.cross_entropy", source_path)
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)
cross_entropy_forward = source.cross_entropy_forward


def main():
    tokens, vocabulary = 8192, 32768
    logits = torch.randn(
        (tokens, vocabulary), device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    target = torch.randint(0, vocabulary, (tokens,), device="cuda", dtype=torch.int64)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    loss, _, accuracy, _, _ = cross_entropy_forward(
        logits,
        target,
        None,
        -100,
        0.0,
        0.0,
        "mean",
        None,
        False,
        True,
        False,
    )
    end.record()
    torch.cuda.synchronize()

    print(f"logits: shape={tuple(logits.shape)}, dtype={logits.dtype}")
    print(f"target: shape={tuple(target.shape)}, vocabulary={vocabulary}")
    print(f"loss={loss.float().item():.6f}, token_accuracy={accuracy.item():.6f}")
    print(f"end_to_end_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
