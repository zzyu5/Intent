import ast
from pathlib import Path

import torch


def load_grouped_gemm():
    source_path = Path(__file__).with_name("08-grouped-gemm.py")
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    tree.body = [node for node in tree.body if node.end_lineno <= 213]
    namespace = {"__file__": str(source_path), "__name__": "triton_grouped_gemm_source"}
    exec(compile(tree, str(source_path), "exec"), namespace)
    return namespace["group_gemm_fn"]


def main():
    # Mixtral 8x7B: 4096 tokens routed top-2, balanced across 8 experts.
    experts, rows_per_expert, hidden, intermediate = 8, 1024, 4096, 14336
    group_a = [
        torch.randn((rows_per_expert, hidden), device="cuda", dtype=torch.float16)
        for _ in range(experts)
    ]
    group_b = [
        torch.randn((hidden, intermediate), device="cuda", dtype=torch.float16)
        for _ in range(experts)
    ]
    grouped_gemm = load_grouped_gemm()

    grouped_gemm(group_a, group_b)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = grouped_gemm(group_a, group_b)
    end.record()
    torch.cuda.synchronize()

    mean = sum(t.float().mean().item() for t in output) / len(output)
    print(f"groups={experts}, A[0]={tuple(group_a[0].shape)}, B[0]={tuple(group_b[0].shape)}, dtype={group_a[0].dtype}")
    print(f"outputs={len(output)}, output[0]={tuple(output[0].shape)}, mean={mean:.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
