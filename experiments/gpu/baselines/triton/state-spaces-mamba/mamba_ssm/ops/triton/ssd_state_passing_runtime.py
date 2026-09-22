import importlib.util
from pathlib import Path

import torch


ROOT = next(parent for parent in Path(__file__).parents if parent.name == "state-spaces-mamba")
SPEC = importlib.util.spec_from_file_location("intent_mamba_runtime", ROOT / "support" / "runtime.py")
RUNTIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME)
RUNTIME.activate_upstream(Path(__file__))


def main():
    from mamba_ssm.ops.triton.ssd_state_passing import _state_passing_fwd

    batch, chunks, heads, head_dim, state_dim = 1, 8, 32, 64, 128
    states = torch.randn((batch, chunks, heads, head_dim * state_dim), device="cuda", dtype=torch.float32) * 0.01
    chunk_decay = -torch.rand((batch, heads, chunks), device="cuda", dtype=torch.float32) * 0.1
    (passed, final), latency = RUNTIME.elapsed_ms(lambda: _state_passing_fwd(states, chunk_decay))
    print(f"B={batch} chunks={chunks} H={heads} state={head_dim}x{state_dim}")
    print(f"passed={tuple(passed.shape)} final={tuple(final.shape)} finite={torch.isfinite(passed).all().item()}")
    print(f"latency_ms={latency:.3f}")


if __name__ == "__main__":
    main()
