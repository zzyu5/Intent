import importlib.util
from pathlib import Path

import torch


SOURCE = Path(__file__).with_name("topk_selector.py")
spec = importlib.util.spec_from_file_location("local_tilelang_topk_selector", SOURCE)
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def main():
    batch, sequence_length, topk = 32, 32768, 2048
    scores = torch.randn(batch, sequence_length, device="cuda", dtype=torch.float32)
    starts = torch.zeros(batch, device="cuda", dtype=torch.int32)
    ends = torch.full((batch,), sequence_length, device="cuda", dtype=torch.int32)

    indices = source.tl_topk(scores, starts, ends, topk)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    indices = source.tl_topk(scores, starts, ends, topk)
    end.record()
    torch.cuda.synchronize()
    selected = torch.gather(scores, 1, indices.to(torch.long))

    print(f"scores={tuple(scores.shape)} starts={tuple(starts.shape)} topk={topk}")
    print(
        f"indices={tuple(indices.shape)} selected_mean={selected.mean().item():.6f} "
        f"selected_min={selected.min().item():.6f}"
    )
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
