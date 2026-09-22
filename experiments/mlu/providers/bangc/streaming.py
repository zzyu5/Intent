import torch

from intent import MojoTarget
from kernels.streaming.attention_f32 import causal_attention_f32, causal_linear_attention_f32
from kernels.streaming.gated_delta import (
    chunk_gated_delta_prepare, chunk_gated_delta_recurrence, recurrent_gated_delta_fwd,
)
from experiments._common.loading import load_module
from experiments._common.model import Tolerance
from .common import RemoteSequence, tilegym_source


def _mojo_reference(context, relative_path, arguments):
    runtime = load_module(context.project_root / relative_path,
        "intent_bangc_mojo_" + relative_path.split("/")[-2])
    source = runtime.prepare(MojoTarget().resolve(), *arguments)

    def reference():
        source.launch()
        return source.result()

    return reference


def causal_attention(context):
    q = torch.randn((8, 128, 32), dtype=torch.float32)
    k, v = torch.randn_like(q), torch.randn_like(q)
    scale = 32 ** -0.5
    sequence = RemoteSequence(context)
    output = sequence.add(causal_attention_f32, {"q": q, "k": k, "v": v, "scale": scale})["output"]
    reference = _mojo_reference(context,
        "experiments/cpu/baselines/mojo/intentdsl/attention/causal/causal_runtime.py", (q, k, v, scale))
    return sequence.comparison(output, reference, Tolerance(2e-4, 1e-5))


def causal_linear_attention(context):
    q = torch.randn((8, 128, 32), dtype=torch.float32)
    k, v = torch.randn_like(q), torch.randn_like(q)
    sequence = RemoteSequence(context)
    outputs = sequence.add(causal_linear_attention_f32, {"q": q, "k": k, "v": v})
    reference = _mojo_reference(context,
        "experiments/cpu/baselines/mojo/intentdsl/attention/linear/linear_runtime.py", (q, k, v))
    return sequence.comparison((outputs["output"], outputs["final_state"]),
        reference, Tolerance(2e-4, 1e-5))


def _gated_delta_inputs():
    batch, length, heads, key_dimension, value_dimension = 2, 2048, 8, 128, 128
    query = torch.randn((batch, length, heads, key_dimension), device="cuda", dtype=torch.bfloat16) * 0.1
    key = torch.randn_like(query) * 0.1
    value = torch.randn((batch, length, heads, value_dimension), device="cuda", dtype=torch.bfloat16) * 0.1
    gate = -torch.rand((batch, length, heads), device="cuda", dtype=torch.bfloat16) * 0.5
    beta = torch.sigmoid(torch.randn_like(gate))
    return {"query": query, "key": key, "value": value, "gate": gate, "beta": beta,
            "scale": key_dimension ** -0.5}


def chunk_gated_delta(context):
    arguments = _gated_delta_inputs()
    sequence = RemoteSequence(context)
    intermediate = sequence.add(chunk_gated_delta_prepare, arguments)
    outputs = sequence.add(chunk_gated_delta_recurrence, intermediate)
    source = tilegym_source(context,
        "experiments/gpu/baselines/cutile/tilegym/scan/gated_delta_chunk/chunk_gated_delta_rule.py",
        "chunk_gated_delta", needs_utils=True)
    tolerance = Tolerance(atol=1e-1, rtol=5e-2)
    return sequence.comparison((outputs["output"], outputs["final_state"]),
        lambda: source.chunk_gated_delta_rule(*(arguments[name] for name in ("query", "key", "value", "gate", "beta")),
            chunk_size=64, initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=False),
        (tolerance, tolerance))


def recurrent_gated_delta(context):
    arguments = _gated_delta_inputs()
    sequence = RemoteSequence(context)
    outputs = sequence.add(recurrent_gated_delta_fwd, arguments, constexprs={"HEAD_GROUP": 1})
    source = tilegym_source(context,
        "experiments/gpu/baselines/cutile/tilegym/scan/gated_delta_recurrent/recurrent_gated_delta_rule.py",
        "recurrent_gated_delta", needs_utils=True)
    tolerance = Tolerance(atol=1e-1, rtol=5e-2)
    return sequence.comparison((outputs["output"], outputs["final_state"]),
        lambda: source.recurrent_gated_delta_rule(*(arguments[name] for name in ("query", "key", "value", "gate", "beta")),
            initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=False),
        (tolerance, tolerance))


CASES = {"causal_attention_f32": causal_attention,
         "causal_linear_attention_f32": causal_linear_attention,
         "chunk_gated_delta": chunk_gated_delta, "recurrent_gated_delta": recurrent_gated_delta}
