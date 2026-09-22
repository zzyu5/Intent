import torch

from kernels.streaming.paged_attention import paged_gqa_decode_partials
from kernels.streaming.splitk_reduce import splitk_attention_f32_to_f16_reduce
from ...loading import load_module
from ...model import Tolerance
from .common import RemoteSequence


def paged_gqa_decode(context):
    batch, query_heads, kv_heads, dimension = 16, 32, 8, 128
    length, page_size, splits = 8192, 16, 8
    pages_per_sequence = length // page_size
    pages = batch * pages_per_sequence
    q = torch.randn((batch, query_heads, dimension), device="cuda", dtype=torch.float16)
    keys = torch.randn((pages, page_size, kv_heads, dimension), device="cuda", dtype=torch.float16)
    values = torch.randn_like(keys)
    indices = torch.arange(pages, device="cuda", dtype=torch.int32)
    offsets = torch.arange(0, pages + 1, pages_per_sequence, device="cuda", dtype=torch.int32)
    split_offsets = torch.arange(0, pages + 1, pages_per_sequence // splits, device="cuda", dtype=torch.int32)
    lengths = torch.full((batch,), length, device="cuda", dtype=torch.int32)
    scale = dimension ** -0.5
    sequence = RemoteSequence(context)
    partials = sequence.add(paged_gqa_decode_partials,
        {"q": q, "key_cache": keys, "value_cache": values, "page_offsets": offsets,
         "page_indices": indices, "sequence_lengths": lengths, "split_offsets": split_offsets, "scale": scale},
        constexprs={"PAGE_SIZE": page_size, "HEAD_GROUP": query_heads // kv_heads, "SPLITS": splits})
    output = sequence.add(splitk_attention_f32_to_f16_reduce,
        {"partial": partials["partial_output"], "partial_lse": partials["partial_lse"]})["output"]
    runtime = load_module(context.project_root /
        "source/triton/vllm/attention/paged_decode/paged_gqa_decode_runtime.py", "intent_bangc_paged_gqa_runtime")
    source = runtime.RUNTIME.load_source(context.project_root /
        "source/triton/vllm/attention/paged_decode/triton_decode_attention.py", "intent_bangc_paged_gqa_source")
    source_output = torch.empty_like(q)
    source_lse = torch.empty((batch, query_heads), device="cuda", dtype=torch.float32)
    workspace = torch.empty((batch, query_heads, splits, dimension + 1), device="cuda", dtype=torch.float32)
    page_table = indices.view(batch, pages_per_sequence)

    def reference():
        source.decode_attention_fwd(q, keys, values, source_output, source_lse,
            page_table, lengths, workspace, splits, scale, page_size=page_size)
        return source_output

    return sequence.comparison(output, reference, Tolerance(atol=5e-2, rtol=5e-2))


CASES = {"paged_gqa_decode": paged_gqa_decode}
