import importlib.util
from pathlib import Path

import torch


ROOT = next(parent for parent in Path(__file__).parents if parent.name == "fla")
SPEC = importlib.util.spec_from_file_location("intent_fla_runtime", ROOT / "support" / "runtime.py")
RUNTIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME)


def main():
    source = RUNTIME.load_causal_conv(Path(__file__).with_name("ops.py"))
    batch, hidden, width = 32, 4096, 4
    x = torch.randn((batch, hidden), device="cuda", dtype=torch.bfloat16)
    cache = torch.randn((batch, hidden, width), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((hidden, width), device="cuda", dtype=torch.float32)
    bias = torch.randn((hidden,), device="cuda", dtype=torch.float32)
    (output, updated_cache), latency = RUNTIME.elapsed_ms(
        lambda: source.causal_conv1d_update(x, cache, weight=weight, bias=bias, activation="silu")
    )
    print(f"decode_batch={batch} hidden={hidden} width={width} dtype={x.dtype}")
    print(f"output={tuple(output.shape)} cache={tuple(updated_cache.shape)} finite={torch.isfinite(output).all().item()}")
    print(f"latency_ms={latency:.3f}")


if __name__ == "__main__":
    main()
