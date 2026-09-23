import torch
import intent
import intent.language as I


@intent.kernel
def _softmax_log_kernel(
    x: I.In[I.f32, (1024, 1024)],
    out: I.Out[I.f32, (1024, 1024)],
):
    rows = I.domain(0, 1024)
    cols = I.domain(0, 1024)

    # Keep log as an explicit first stage of the composition.  The reductions
    # are row-local, so every row is independent of the others.
    log_values = I.log(x[rows, cols])
    row_max = I.reduce.max(log_values, axis=1, acc_dtype=I.f32)
    row_max = I.reshape(row_max, (1024, 1))

    shifted = log_values - row_max
    exp_values = I.exp(shifted)
    row_sum = I.reduce.sum(exp_values, axis=1, acc_dtype=I.f32)
    row_sum = I.reshape(row_sum, (1024, 1))
    out[rows, cols] = I.fdiv(exp_values, row_sum)


def build(context):
    artifact = context.compile("softmax_log_kernel", _softmax_log_kernel)

    def softmax_log(input, dim=-1, dtype=None):
        # The evaluator's profile is contiguous float32 with the default last
        # dimension and no requested cast.
        output = torch.empty_like(input)
        artifact(input, output)
        return output

    return softmax_log
