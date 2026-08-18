import importlib.util
from pathlib import Path

import torch


ROOT = next(parent for parent in Path(__file__).parents if parent.name == "state-spaces-mamba")
SPEC = importlib.util.spec_from_file_location("intent_mamba_runtime", ROOT / "support" / "runtime.py")
RUNTIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME)
RUNTIME.activate_upstream(Path(__file__))


def main():
    from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_fwd

    batch, sequence, heads, head_dim = 1, 2048, 32, 64
    groups, state_dim, chunk = 8, 128, 256
    chunks = (sequence + chunk - 1) // chunk
    x = torch.randn((batch, sequence, heads, head_dim), device="cuda", dtype=torch.bfloat16)
    c = torch.randn((batch, sequence, groups, state_dim), device="cuda", dtype=torch.bfloat16)
    cb = torch.randn((batch, chunks, groups, chunk, chunk), device="cuda", dtype=torch.bfloat16) * 0.01
    dt = torch.rand((batch, heads, chunks, chunk), device="cuda", dtype=torch.float32) * 0.01
    decay = -torch.rand_like(dt).cumsum(-1) * 0.01
    previous = torch.randn((batch, chunks, heads, head_dim, state_dim), device="cuda", dtype=torch.float32) * 0.01
    residual = torch.randn((heads,), device="cuda", dtype=torch.float32) * 0.01
    (output, _), latency = RUNTIME.elapsed_ms(
        lambda: _chunk_scan_fwd(cb, x, dt, decay, c, previous, D=residual)
    )
    print(f"B={batch} S={sequence} H={heads} P={head_dim} G={groups} N={state_dim} chunk={chunk}")
    print(f"output={tuple(output.shape)} dtype={output.dtype} finite={torch.isfinite(output).all().item()}")
    print(f"latency_ms={latency:.3f}")


if __name__ == "__main__":
    main()
