import importlib.util
import sys
from pathlib import Path

import torch


upstream_root = Path(__file__).parents[2]
sys.path.insert(0, str(upstream_root / "support"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load(
    "liger_kernel.ops.cross_entropy",
    upstream_root / "loss" / "cross_entropy" / "cross_entropy.py",
)
source = load(
    "liger_kernel.ops.fused_linear_cross_entropy",
    Path(__file__).with_name("fused_linear_cross_entropy.py"),
)
fused_linear_cross_entropy_forward = source.fused_linear_cross_entropy_forward


def main():
    tokens, hidden, vocabulary = 2048, 4096, 32768
    hidden_states = torch.randn(
        (tokens, hidden), device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    lm_head = torch.randn(
        (vocabulary, hidden), device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    target = torch.randint(0, vocabulary, (tokens,), device="cuda", dtype=torch.int64)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    loss, _, accuracy, _, _, _, _ = fused_linear_cross_entropy_forward(
        hidden_states,
        lm_head,
        target,
        reduction="mean",
        return_token_accuracy=True,
    )
    end.record()
    torch.cuda.synchronize()

    print(f"hidden states: shape={tuple(hidden_states.shape)}, dtype={hidden_states.dtype}")
    print(f"LM head: shape={tuple(lm_head.shape)}, vocabulary={vocabulary}")
    print(f"loss={loss.float().item():.6f}, token_accuracy={accuracy.item():.6f}")
    print(f"end_to_end_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
