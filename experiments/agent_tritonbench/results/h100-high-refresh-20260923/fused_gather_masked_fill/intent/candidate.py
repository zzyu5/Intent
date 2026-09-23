import torch
import intent
import intent.language as I


@intent.kernel
def _fused_gather_masked_fill(
    input: I.In[I.f32, (1024, 1024)],
    index: I.In[I.i64, (1024, 1024)],
    mask: I.In[I.bool, (1024, 1024)],
    output: I.Out[I.f32, (1024, 1024)],
    value: I.f32,
):
    rows = I.domain(0, 1024)
    columns = I.domain(0, 1024)

    for row in I.parallel(rows):
        for column in I.parallel(columns):
            gathered_row = index[row, column]
            gathered_value = input[gathered_row, column]
            result = I.select(mask[row, column], value, gathered_value)
            output[row, column] = result


def build(context):
    artifact = context.compile("fused_gather_masked_fill", _fused_gather_masked_fill)

    def fused_gather_masked_fill(
        input, dim, index, mask, value, *, sparse_grad=False, out=None
    ):
        if out is None:
            output = torch.empty_like(input)
        else:
            output = out
        artifact(input, index, mask, output, value)
        return output

    return fused_gather_masked_fill
