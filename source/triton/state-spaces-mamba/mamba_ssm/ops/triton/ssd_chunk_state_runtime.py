import importlib.util
from pathlib import Path

import torch


ROOT = next(parent for parent in Path(__file__).parents if parent.name == "state-spaces-mamba")
SPEC = importlib.util.spec_from_file_location("intent_mamba_runtime", ROOT / "support" / "runtime.py")
RUNTIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME)
RUNTIME.activate_upstream(Path(__file__))


def main():
    from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_state_fwd

    batch, sequence, heads, head_dim = 1, 2048, 32, 64
    groups, state_dim, chunk = 8, 128, 256
    chunks = (sequence + chunk - 1) // chunk
    x = torch.randn((batch, sequence, heads, head_dim), device="cuda", dtype=torch.bfloat16)
    state_input = torch.randn((batch, sequence, groups, state_dim), device="cuda", dtype=torch.bfloat16)
    dt = torch.rand((batch, heads, chunks, chunk), device="cuda", dtype=torch.float32) * 0.01
    decay = -torch.rand_like(dt).cumsum(-1) * 0.01
    states, latency = RUNTIME.elapsed_ms(lambda: _chunk_state_fwd(state_input, x, dt, decay, states_in_fp32=True))
    print(f"B={batch} S={sequence} H={heads} P={head_dim} G={groups} N={state_dim} chunk={chunk}")
    print(f"states={tuple(states.shape)} dtype={states.dtype} finite={torch.isfinite(states).all().item()}")
    print(f"latency_ms={latency:.3f}")


if __name__ == "__main__":
    main()
