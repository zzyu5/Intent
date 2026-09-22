import importlib.util
from pathlib import Path

import torch


def _runtime_module():
    path = Path(__file__).parents[2] / "support" / "runtime.py"
    spec = importlib.util.spec_from_file_location("intent_meta_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _balanced_metadata(topk_ids, block_m, experts):
    flat = topk_ids.reshape(-1)
    order = torch.argsort(flat)
    pieces = []
    block_experts = []
    sentinel = flat.numel()
    for expert in range(experts):
        ids = order[flat[order] == expert].to(torch.int32)
        pad = (-ids.numel()) % block_m
        if pad:
            ids = torch.cat((ids, torch.full((pad,), sentinel, device=ids.device, dtype=torch.int32)))
        pieces.append(ids)
        block_experts.extend([expert] * (ids.numel() // block_m))
    sorted_ids = torch.cat(pieces)
    expert_ids = torch.tensor(block_experts, device=topk_ids.device, dtype=torch.int32)
    num_tokens = torch.tensor([sorted_ids.numel()], device=topk_ids.device, dtype=torch.int32)
    return sorted_ids, expert_ids, num_tokens


def run(source_path: Path, variant: str):
    runtime = _runtime_module()
    source = runtime.load_source(source_path, f"intent_meta_moe_{variant}", stub_vllm=True)
    tokens, hidden, output, experts, topk = 2048, 4096, 14336, 8, 2
    x = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((experts, output, hidden), device="cuda", dtype=torch.bfloat16)
    ids = torch.stack((torch.arange(tokens, device="cuda") % experts, (torch.arange(tokens, device="cuda") + 1) % experts), dim=1).to(torch.int32)
    routed_weight = torch.full((tokens, topk), 0.5, device="cuda", dtype=torch.float32)
    if variant == "grouped":
        config = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}
        block_m = config["BLOCK_SIZE_M"]
    elif variant == "splitk":
        config = {"block_m": 32, "block_n": 64, "block_k": 64, "group_m": 8, "split_k": 2}
        block_m = config["block_m"]
    elif variant == "column_major":
        config = {"block_m": 64, "block_n": 64, "block_k": 32}
        block_m = config["block_m"]
    else:
        raise ValueError(variant)
    sorted_ids, expert_ids, padded = _balanced_metadata(ids, block_m, experts)
    result = torch.zeros((tokens, topk, output), device="cuda", dtype=x.dtype)

    def call():
        if variant == "splitk":
            result.zero_()
        source.invoke_fused_moe_kernel(x, weight, result, routed_weight, ids, sorted_ids, expert_ids, padded, False, topk, config)
        return result

    value, latency = runtime.elapsed_ms(call)
    print(f"algorithm=moe_{variant}_expert_projection tokens={tokens} hidden={hidden} output={output} experts={experts} topk={topk}")
    print(f"output={tuple(value.shape)} dtype={value.dtype} mean={value.float().mean().item():.6f}")
    print(f"latency_ms={latency:.3f}")
