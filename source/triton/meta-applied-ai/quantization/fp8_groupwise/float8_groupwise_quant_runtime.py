import importlib.util
import os
from pathlib import Path

import torch


def main():
    os.environ["TRITON_ALLOW_NON_CONSTEXPR_GLOBALS"] = "1"
    root = next(parent for parent in Path(__file__).parents if parent.name == "meta-applied-ai")
    helper_path = root / "support" / "runtime.py"
    spec = importlib.util.spec_from_file_location("intent_meta_runtime", helper_path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    source = runtime.load_source(Path(__file__).with_name("float8_groupwise_quant.py"), "intent_fp8_groupwise")
    x = torch.randn((8192, 4096), device="cuda", dtype=torch.bfloat16)
    (quantized, scales), latency = runtime.elapsed_ms(lambda: source.float8_groupwise_quantize(x, 128))
    print(f"input={tuple(x.shape)} dtype={x.dtype} group=128")
    print(f"output={tuple(quantized.shape)} scales={tuple(scales.shape)}")
    print(f"latency_ms={latency:.3f}")


if __name__ == "__main__":
    main()
