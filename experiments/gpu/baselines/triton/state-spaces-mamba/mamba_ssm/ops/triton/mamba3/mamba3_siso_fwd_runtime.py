import importlib.util
from pathlib import Path

import torch


ROOT = next(parent for parent in Path(__file__).parents if parent.name == "state-spaces-mamba")
SPEC = importlib.util.spec_from_file_location("intent_mamba_runtime", ROOT / "support" / "runtime.py")
RUNTIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME)
RUNTIME.activate_upstream(Path(__file__))


def main():
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_fwd import mamba3_siso_fwd

    batch, sequence, qk_heads, heads = 1, 2048, 4, 16
    qk_dim, value_dim, angle_dim = 32, 64, 16
    dtype = torch.bfloat16
    q = torch.randn((batch, sequence, qk_heads, qk_dim), device="cuda", dtype=dtype) * 0.1
    k = torch.randn_like(q) * 0.1
    v = torch.randn((batch, sequence, heads, value_dim), device="cuda", dtype=dtype) * 0.1
    adt = -torch.rand((batch, heads, sequence), device="cuda", dtype=torch.float32) * 0.1
    dt = torch.rand((batch, heads, sequence), device="cuda", dtype=torch.float32) * 0.1
    trap = torch.rand((batch, heads, sequence), device="cuda", dtype=torch.float32) * 0.1
    q_bias = torch.randn((heads, qk_dim), device="cuda", dtype=dtype) * 0.01
    k_bias = torch.randn_like(q_bias) * 0.01
    angles = torch.randn((batch, sequence, heads, angle_dim), device="cuda", dtype=torch.float32) * 0.01
    residual = torch.randn((heads,), device="cuda", dtype=torch.float32) * 0.01
    gate = torch.randn((batch, sequence, heads, value_dim), device="cuda", dtype=dtype) * 0.1
    result, latency = RUNTIME.elapsed_ms(
        lambda: mamba3_siso_fwd(q, k, v, adt, dt, trap, q_bias, k_bias, angles, D=residual, Z=gate, chunk_size=64)
    )
    output = result[0]
    print(f"B={batch} S={sequence} HQK={qk_heads} H={heads} DQK={qk_dim} DV={value_dim}")
    print(f"output={tuple(output.shape)} dtype={output.dtype} finite={torch.isfinite(output).all().item()}")
    print(f"latency_ms={latency:.3f}")


if __name__ == "__main__":
    main()
