import importlib.util
import sys
from pathlib import Path

import tilegym
import torch


SOURCE = Path(__file__).with_name("moe.py")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load(
    "tilegym.ops.cutile.utils",
    Path(__file__).parents[2] / "support" / "utils.py",
)
align = load(
    "tilegym.ops.cutile._local_moe_align_for_moe",
    Path(__file__).parents[1] / "alignment" / "moe_align_block.py",
)
source = load("tilegym.ops.cutile._local_moe", SOURCE)


def main():
    tokens, hidden_size, intermediate_size = 4096, 4096, 14336
    experts, top_k, block_size = 8, 2, 128
    hidden = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    expert_weights = torch.randn(
        experts, intermediate_size, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    topk_ids = (
        torch.arange(tokens * top_k, device="cuda", dtype=torch.long)
        .remainder(experts)
        .reshape(tokens, top_k)
        .contiguous()
    )
    topk_weights = torch.softmax(
        torch.randn(tokens, top_k, device="cuda", dtype=torch.float32), dim=-1
    ).to(torch.bfloat16).contiguous()
    sorted_ids, expert_ids, padded_tokens, _, _ = align.moe_align_block_size(
        topk_ids, block_size, experts
    )
    output = torch.empty(
        tokens, top_k, intermediate_size, device="cuda", dtype=torch.bfloat16
    )
    config = {
        "TILE_SIZE_M": 128,
        "TILE_SIZE_N": 128,
        "TILE_SIZE_K": 64,
        "GROUP_SIZE_M": 32,
        "num_warps": 8,
        "num_stages": 4,
    }

    def launch():
        source.invoke_fused_moe_kernel(
            hidden,
            expert_weights,
            output,
            None,
            None,
            topk_weights,
            topk_ids,
            sorted_ids,
            expert_ids,
            padded_tokens,
            False,
            top_k,
            config,
            torch.bfloat16,
            False,
        )

    launch()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    launch()
    end.record()
    torch.cuda.synchronize()

    print(
        f"hidden={tuple(hidden.shape)} experts={tuple(expert_weights.shape)} "
        f"topk_ids={tuple(topk_ids.shape)} dtype={hidden.dtype}"
    )
    print(f"output={tuple(output.shape)} mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
