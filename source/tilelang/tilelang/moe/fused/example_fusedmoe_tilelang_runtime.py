import importlib.util
import sys
from pathlib import Path

import torch


def load_source():
    source_path = Path(__file__).with_name("example_fusedmoe_tilelang.py")
    sys.path.insert(0, str(source_path.parent))
    spec = importlib.util.spec_from_file_location("tilelang_fused_moe_source", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    # DeepSeek-V3 hidden/expert widths with a memory-feasible eight-expert shard.
    source = load_source()
    data = source.generate_input(
        dhidden=7168,
        dexpert=2048,
        nroutedexperts=8,
        nsharedexperts=1,
        nexpertspertoken=4,
        bs=1,
        seqlen=8192,
        seed=81394,
    )
    input_tensor, weights, config = data

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = source.custom_kernel(data)
    end.record()
    torch.cuda.synchronize()

    print(f"input: shape={tuple(input_tensor.shape)}, dtype={input_tensor.dtype}")
    print(
        f"experts={config['n_routed_experts']}, shared={config['n_shared_experts']}, "
        f"top_k={config['n_experts_per_token']}, expert_dim={config['d_expert']}"
    )
    print(f"weight tensors={len(weights)}, output={tuple(output.shape)}, mean={output.float().mean().item():.6f}")
    print(f"end_to_end_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
