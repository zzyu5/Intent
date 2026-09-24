import torch
import intent
import intent.language as I


# The fixed profile has one million input elements.  Logical groups retain
# independent compaction work while keeping the prefix state small.
GROUPS = 4096
CHUNK = 256
OUTPUT_SIZE = 524236


@intent.fn
def _gelu_none(value: I.f32) -> I.f32:
    half = I.cast(0.5, I.f32)
    one = I.cast(1.0, I.f32)
    inv_sqrt_two = I.cast(0.7071067811865476, I.f32)
    return half * value * (one + I.erf(value * inv_sqrt_two))


@intent.fn
def _gelu_tanh(value: I.f32) -> I.f32:
    half = I.cast(0.5, I.f32)
    one = I.cast(1.0, I.f32)
    scale = I.cast(0.7978845608028654, I.f32)
    coefficient = I.cast(0.044715, I.f32)
    cubic = value * value * value
    argument = scale * (value + coefficient * cubic)
    return half * value * (one + I.tanh(argument, approximate=True))


@intent.kernel
def _count_groups(
    mask: I.In[I.bool, ("N",)],
    counts: I.Out[I.i32, ("G",)],
):
    groups = I.domain(0, GROUPS)
    lanes = I.domain(0, CHUNK)
    group_ids = I.indices(groups)
    lane_ids = I.indices(lanes)
    group_ids = I.reshape(group_ids, (GROUPS, 1))
    lane_ids = I.reshape(lane_ids, (1, CHUNK))
    positions = group_ids * CHUNK + lane_ids
    selected = I.cast(mask[positions], I.i32)
    counts[groups] = I.reduce.sum(selected, axis=1, acc_dtype=I.i32)


@intent.kernel
def _exclusive_group_offsets(
    counts: I.In[I.i32, ("G",)],
    offsets: I.Out[I.i32, ("G",)],
):
    groups = I.domain(0, GROUPS)
    offsets[groups] = I.cumsum(
        counts[groups], axis=0, inclusive=False, acc_dtype=I.i32
    )


@intent.kernel
def _compact_gelu_none(
    input: I.In[I.f32, ("N",)],
    mask: I.In[I.bool, ("N",)],
    offsets: I.In[I.i32, ("G",)],
    out: I.Out[I.f32, ("K",)],
    other: I.f32,
    alpha: I.f32,
):
    groups = I.domain(0, GROUPS)
    for group in I.parallel(groups):
        base = group * CHUNK
        rank = I.cast(0, I.index)
        for lane in range(CHUNK):
            position = base + lane
            if mask[position]:
                output_position = I.cast(offsets[group], I.index) + rank
                value = input[position] + alpha * other
                out[output_position] = _gelu_none(value)
                rank = rank + I.cast(1, I.index)


@intent.kernel
def _compact_gelu_tanh(
    input: I.In[I.f32, ("N",)],
    mask: I.In[I.bool, ("N",)],
    offsets: I.In[I.i32, ("G",)],
    out: I.Out[I.f32, ("K",)],
    other: I.f32,
    alpha: I.f32,
):
    groups = I.domain(0, GROUPS)
    for group in I.parallel(groups):
        base = group * CHUNK
        rank = I.cast(0, I.index)
        for lane in range(CHUNK):
            position = base + lane
            if mask[position]:
                output_position = I.cast(offsets[group], I.index) + rank
                value = input[position] + alpha * other
                out[output_position] = _gelu_tanh(value)
                rank = rank + I.cast(1, I.index)


def build(context):
    count_groups = context.compile("masked_select_count", _count_groups)
    make_offsets = context.compile("masked_select_offsets", _exclusive_group_offsets)
    compact_none = context.compile("masked_select_add_gelu_none", _compact_gelu_none)
    compact_tanh = context.compile("masked_select_add_gelu_tanh", _compact_gelu_tanh)

    def fused_masked_select_add_gelu(
        input, mask, other, *, alpha=1, approximate="none", out=None
    ):
        if out is None:
            out = torch.empty(
                (OUTPUT_SIZE,), dtype=input.dtype, device=input.device
            )
        counts = torch.empty(
            (GROUPS,), dtype=torch.int32, device=input.device
        )
        offsets = torch.empty(
            (GROUPS,), dtype=torch.int32, device=input.device
        )

        count_groups(mask, counts)
        make_offsets(counts, offsets)

        other_value = float(other)
        alpha_value = float(alpha)
        if approximate == "tanh":
            compact_tanh(
                input, mask, offsets, out, other_value, alpha_value
            )
        else:
            compact_none(
                input, mask, offsets, out, other_value, alpha_value
            )
        return out

    return fused_masked_select_add_gelu
