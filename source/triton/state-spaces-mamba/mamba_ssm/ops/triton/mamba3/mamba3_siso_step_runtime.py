import importlib.util
from pathlib import Path

import torch


ROOT = next(parent for parent in Path(__file__).parents if parent.name == "state-spaces-mamba")
SPEC = importlib.util.spec_from_file_location("intent_mamba_runtime", ROOT / "support" / "runtime.py")
RUNTIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME)
RUNTIME.activate_upstream(Path(__file__))


def main():
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_step import mamba3_siso_step

    batch, qk_heads, heads = 32, 4, 16
    qk_dim, value_dim, angle_dim = 32, 64, 16
    dtype = torch.bfloat16
    q = torch.randn((batch, qk_heads, qk_dim), device="cuda", dtype=dtype) * 0.1
    k = torch.randn_like(q) * 0.1
    v = torch.randn((batch, heads, value_dim), device="cuda", dtype=dtype) * 0.1
    adt = -torch.rand((batch, heads), device="cuda", dtype=torch.float32) * 0.1
    dt = torch.rand((batch, heads), device="cuda", dtype=torch.float32) * 0.1
    trap = torch.rand((batch, heads), device="cuda", dtype=torch.float32) * 0.1
    q_bias = torch.randn((heads, qk_dim), device="cuda", dtype=dtype) * 0.01
    k_bias = torch.randn_like(q_bias) * 0.01
    angles = torch.randn((batch, heads, angle_dim), device="cuda", dtype=torch.float32) * 0.01
    residual = torch.randn((heads,), device="cuda", dtype=torch.float32) * 0.01
    gate = torch.randn((batch, heads, value_dim), device="cuda", dtype=dtype) * 0.1
    states = (
        torch.zeros((batch, heads, angle_dim), device="cuda", dtype=torch.float32),
        torch.zeros((batch, heads, value_dim, qk_dim), device="cuda", dtype=torch.float32),
        torch.zeros((batch, heads, qk_dim), device="cuda", dtype=torch.float32),
        torch.zeros((batch, heads, value_dim), device="cuda", dtype=torch.float32),
    )
    (output, output_states), latency = RUNTIME.elapsed_ms(
        lambda: mamba3_siso_step(q, k, v, adt, dt, trap, q_bias, k_bias, angles, D=residual, Z=gate, Input_States=states)
    )
    print(f"B={batch} HQK={qk_heads} H={heads} DQK={qk_dim} DV={value_dim}")
    print(f"output={tuple(output.shape)} state_shapes={[tuple(value.shape) for value in output_states]}")
    print(f"finite={torch.isfinite(output).all().item()} latency_ms={latency:.3f}")


if __name__ == "__main__":
    main()
