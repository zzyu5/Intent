import importlib.util
from pathlib import Path

import torch


def main():
    root = next(parent for parent in Path(__file__).parents if parent.name == "meta-applied-ai")
    spec = importlib.util.spec_from_file_location("intent_meta_runtime", root / "support" / "runtime.py")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    source = runtime.load_source(Path(__file__).with_name("scaled_fp8_gemm.py"), "intent_scaled_fp8_gemm")
    m, n, k = 4096, 14336, 4096
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    output, latency = runtime.elapsed_ms(lambda: source.scaled_mm_splitk(a, b, 1.0, 1.0))
    print(f"A={tuple(a.shape)} B={tuple(b.shape)} dtype={a.dtype}")
    print(f"output={tuple(output.shape)} dtype={output.dtype} mean={output.float().mean().item():.6f}")
    print(f"latency_ms={latency:.3f}")


if __name__ == "__main__":
    main()
