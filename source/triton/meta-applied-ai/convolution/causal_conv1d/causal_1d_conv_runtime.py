import importlib.util
from pathlib import Path

import torch


def main():
    root = next(parent for parent in Path(__file__).parents if parent.name == "meta-applied-ai")
    spec = importlib.util.spec_from_file_location("intent_meta_runtime", root / "support" / "runtime.py")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    source = runtime.load_source(Path(__file__).with_name("causal_1d_conv.py"), "intent_causal_conv1d")
    batch, channels, sequence, width = 4, 4096, 4096, 4
    x = torch.randn((batch, sequence, channels), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    weight = torch.randn((channels, width), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((channels,), device="cuda", dtype=torch.bfloat16)
    output, latency = runtime.elapsed_ms(lambda: source.causal_conv1d_fwd(x, weight, bias=bias, activation="silu"))
    print(f"x={tuple(x.shape)} width={width} dtype={x.dtype} activation=silu")
    print(f"output={tuple(output.shape)} mean={output.float().mean().item():.6f}")
    print(f"latency_ms={latency:.3f}")


if __name__ == "__main__":
    main()
