import torch
import intent
import intent.language as I


ELEMENTS = 1048576
GROUPS = 1024
SEGMENT = ELEMENTS // GROUPS


@intent.kernel
def prefix_partials(
    mask: I.In[I.bool, (ELEMENTS,)],
    prefixes: I.Out[I.i32, (ELEMENTS,)],
    counts: I.Out[I.i32, (GROUPS,)],
):
    elements = I.domain(0, ELEMENTS)
    for group in I.parallel(I.domain(0, GROUPS)):
        begin = group * SEGMENT
        segment = elements[begin:begin + SEGMENT]
        local = I.cumsum(mask[segment], axis=0, acc_dtype=I.i32)
        prefixes[segment] = local
        counts[group] = local[SEGMENT - 1]


@intent.kernel
def prefix_offsets(
    counts: I.In[I.i32, (GROUPS,)],
    offsets: I.Out[I.i32, (GROUPS,)],
):
    values = counts[:]
    offsets[:] = I.cumsum(values, axis=0) - values


@intent.kernel
def select_gelu(
    input: I.In[I.f32, (ELEMENTS,)],
    mask: I.In[I.bool, (ELEMENTS,)],
    prefixes: I.In[I.i32, (ELEMENTS,)],
    offsets: I.In[I.i32, (GROUPS,)],
    output: I.Out[I.f32, (524236,)],
    other: I.f32,
    alpha: I.f32,
    APPROXIMATE: I.Constexpr[bool],
):
    for source in I.parallel(I.domain(0, ELEMENTS)):
        if mask[source]:
            group = source // SEGMENT
            destination = I.cast(prefixes[source] + offsets[group] - 1, I.index)
            I.assume_in_bounds(destination, output, axis=0)
            value = input[source] + alpha * other
            if APPROXIMATE:
                argument = 0.7978845608028654 * (value + 0.044715 * value * value * value)
                result = 0.5 * value * (1.0 + I.tanh(argument, approximate=True))
            else:
                result = 0.5 * value * (1.0 + I.erf(value * 0.7071067811865476))
            output[destination] = result


def build(context):
    partials = context.compile("prefix_partials", prefix_partials)
    offsets_kernel = context.compile("prefix_offsets", prefix_offsets)
    exact = context.compile("select_gelu_exact", select_gelu, constexprs={"APPROXIMATE": False})
    tanh = context.compile("select_gelu_tanh", select_gelu, constexprs={"APPROXIMATE": True})

    def fused_masked_select_add_gelu(input, mask, other, *, alpha=1, approximate="none", out=None):
        prefixes = torch.empty((ELEMENTS,), dtype=torch.int32, device=input.device)
        counts = torch.empty((GROUPS,), dtype=torch.int32, device=input.device)
        offsets = torch.empty_like(counts)
        output = out if out is not None else torch.empty((524236,), dtype=input.dtype, device=input.device)
        partials(mask, prefixes, counts)
        offsets_kernel(counts, offsets)
        consumer = exact if approximate == "none" else tanh
        consumer(input, mask, prefixes, offsets, output, float(other), float(alpha))
        return output

    return fused_masked_select_add_gelu
