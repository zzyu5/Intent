import intent
import intent.language as I


TOKENS = 32768
VOCABULARY = 8192
FEATURES = 1021


@intent.kernel
def embedding_forward_lookup(
    embedding_table: I.In[I.f32, ("V", "D")],
    indices: I.In[I.i32, ("M",)],
    output: I.Out[I.f32, ("M", "D")],
):
    M = indices.shape[0]
    D = embedding_table.shape[1]
    features = I.domain(0, D)
    for token_region in I.parallel(
        I.partition(I.domain(0, M), extent=I.auto("M_TILE"))
    ):
        rows = indices[token_region]
        I.assume_in_bounds(rows, embedding_table, axis=0)
        output[token_region, features] = embedding_table[rows, features]


@intent.kernel
def embedding_backward_atomic(
    indices: I.In[I.i32, ("M",)],
    grad_output: I.In[I.f32, ("M", "D")],
    grad_weight: I.InOut[I.f32, ("V", "D")],
):
    M, D = grad_output.shape
    features = I.domain(0, D)
    for token in I.parallel(I.domain(0, M)):
        embedding = indices[token]
        I.assume_in_bounds(embedding, grad_weight, axis=0)
        I.atomic_add(
            grad_weight,
            index=(embedding, features),
            value=grad_output[token, features],
        )
