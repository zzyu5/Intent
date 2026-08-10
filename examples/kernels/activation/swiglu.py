import intent
import intent.language as I


TOKENS = 4096
FEATURES = 14336


@intent.kernel
def swiglu_forward(
    gate: I.In[I.bf16, ("M", "N")],
    up: I.In[I.bf16, ("M", "N")],
    output: I.Out[I.bf16, ("M", "N")],
):
    M, N = gate.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        gate_values = I.cast(gate[row, columns], I.f32)
        up_values = I.cast(up[row, columns], I.f32)
        sigmoid = 1.0 / (1.0 + I.exp(-gate_values))
        output[row, columns] = I.cast(
            gate_values * sigmoid * up_values,
            I.bf16,
        )
