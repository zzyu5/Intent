import importlib.util
from pathlib import Path

import torch


ROOT = next(parent for parent in Path(__file__).parents if parent.name == "fla")
SPEC = importlib.util.spec_from_file_location("intent_fla_runtime", ROOT / "support" / "runtime.py")
RUNTIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME)


def main():
    source = RUNTIME.load_causal_conv(Path(__file__).with_name("ops.py"))
    lengths = [2048, 1536, 1024, 512]
    offsets = torch.tensor([0, 2048, 3584, 4608, 5120], device="cuda", dtype=torch.int64)
    chunk_size, hidden, width = 64, 4096, 4
    x = torch.randn((1, sum(lengths), hidden), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((hidden, width), device="cuda", dtype=torch.float32)
    bias = torch.randn((hidden,), device="cuda", dtype=torch.float32)
    chunk_indices = RUNTIME._prepare_chunk_indices(offsets, chunk_size)
    (output, final_state), latency = RUNTIME.elapsed_ms(
        lambda: source.causal_conv1d_fwd(
            x, weight, bias, None, output_final_state=True, activation="silu",
            cu_seqlens=offsets, chunk_indices=chunk_indices, BT=chunk_size,
        )
    )
    print(f"packed_lengths={lengths} hidden={hidden} width={width} dtype={x.dtype}")
    print(f"output={tuple(output.shape)} final_state={tuple(final_state.shape)} finite={torch.isfinite(output).all().item()}")
    print(f"latency_ms={latency:.3f}")


if __name__ == "__main__":
    main()
