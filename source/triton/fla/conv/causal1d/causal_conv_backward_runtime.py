import importlib.util
from pathlib import Path

import torch


ROOT = next(parent for parent in Path(__file__).parents if parent.name == "fla")
SPEC = importlib.util.spec_from_file_location("intent_fla_runtime", ROOT / "support" / "runtime.py")
RUNTIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME)


def main():
    source = RUNTIME.load_causal_conv(Path(__file__).with_name("ops.py"))
    batch, sequence, hidden, width = 2, 2048, 4096, 4
    x = torch.randn((batch, sequence, hidden), device="cuda", dtype=torch.bfloat16)
    grad = torch.randn_like(x)
    weight = torch.randn((hidden, width), device="cuda", dtype=torch.float32)
    bias = torch.randn((hidden,), device="cuda", dtype=torch.float32)
    result, latency = RUNTIME.elapsed_ms(
        lambda: source.causal_conv1d_bwd(x, grad, None, weight=weight, bias=bias, activation="silu")
    )
    dx, dw, db, _, _ = result
    print(f"B={batch} S={sequence} hidden={hidden} width={width} dtype={x.dtype}")
    print(f"dx={tuple(dx.shape)} dw={tuple(dw.shape)} db={tuple(db.shape)} finite={torch.isfinite(dx).all().item()}")
    print(f"latency_ms={latency:.3f}")


if __name__ == "__main__":
    main()
