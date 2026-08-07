import importlib.util
import sys
from pathlib import Path

import tilegym
import torch


SOURCE = Path(__file__).with_name("moe_align_block.py")
UTILS = Path(__file__).parents[2] / "support" / "utils.py"
utils_spec = importlib.util.spec_from_file_location("tilegym.ops.cutile.utils", UTILS)
utils = importlib.util.module_from_spec(utils_spec)
sys.modules[utils_spec.name] = utils
utils_spec.loader.exec_module(utils)
spec = importlib.util.spec_from_file_location("tilegym.ops.cutile._local_moe_align", SOURCE)
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)


def main():
    tokens, top_k, experts, block_size = 8192, 8, 64, 128
    topk_ids = torch.randint(
        0, experts, (tokens, top_k), device="cuda", dtype=torch.long
    )

    result = source.moe_align_block_size(topk_ids, block_size, experts)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = source.moe_align_block_size(topk_ids, block_size, experts)
    end.record()
    torch.cuda.synchronize()
    sorted_ids, expert_ids, padded_tokens, cumsum, max_expert_count = result

    print(
        f"topk_ids={tuple(topk_ids.shape)} experts={experts} block_size={block_size}"
    )
    print(
        f"sorted_ids={tuple(sorted_ids.shape)} expert_ids={tuple(expert_ids.shape)} "
        f"padded_tokens={padded_tokens.item()} cumsum={tuple(cumsum.shape)} "
        f"max_expert_count={max_expert_count.item()}"
    )
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
