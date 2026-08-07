import importlib.util
import sys
from pathlib import Path

import tilegym
import torch


SOURCE = Path(__file__).with_name("rope.py")
UTILS = Path(__file__).parents[2] / "support" / "utils.py"
utils_spec = importlib.util.spec_from_file_location("tilegym.ops.cutile.utils", UTILS)
utils = importlib.util.module_from_spec(utils_spec)
sys.modules[utils_spec.name] = utils
utils_spec.loader.exec_module(utils)
spec = importlib.util.spec_from_file_location("tilegym.ops.cutile._local_rope", SOURCE)
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)


def main():
    batch, q_heads, kv_heads, seqlen, head_dim = 2, 32, 8, 4096, 128
    q = torch.randn(batch, q_heads, seqlen, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, kv_heads, seqlen, head_dim, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(seqlen, device="cuda", dtype=torch.float32)
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2, device="cuda", dtype=torch.float32) / head_dim)
    )
    half_angles = torch.outer(positions, inv_freq)
    angles = torch.cat((half_angles, half_angles), dim=-1)
    cos = angles.cos().unsqueeze(0).to(torch.bfloat16)
    sin = angles.sin().unsqueeze(0).to(torch.bfloat16)

    q_warm, k_warm = q.clone(), k.clone()
    source.apply_rope_base(q_warm, k_warm, cos, sin)
    torch.cuda.synchronize()
    q_timed, k_timed = q.clone(), k.clone()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    q_output, k_output = source.apply_rope_base(q_timed, k_timed, cos, sin)
    end.record()
    torch.cuda.synchronize()

    print(
        f"q={tuple(q.shape)} k={tuple(k.shape)} cos={tuple(cos.shape)} dtype={q.dtype}"
    )
    print(
        f"q_output={tuple(q_output.shape)} k_output={tuple(k_output.shape)} "
        f"mean={q_output.float().mean().item():.6f}"
    )
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
