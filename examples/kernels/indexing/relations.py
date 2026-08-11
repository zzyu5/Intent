import intent
import intent.language as I


OFFSET_ROWS = 4096
OFFSET_FEATURES = 1024
GQA_QUERY_HEADS = 32
GQA_KEY_HEADS = 8
GQA_TOKENS = 4096
GQA_HEAD_GROUP = GQA_QUERY_HEADS // GQA_KEY_HEADS
LOOKUP_ROWS = 65536
LOOKUP_ENTRIES = 65536
LOOKUP_FEATURES = 128


@intent.kernel
def shifted_row_copy(
    x: I.In[I.f32, ("M", "N")],
    output: I.Out[I.f32, ("M", "N")],
):
    M, N = x.shape
    features = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        shifted_row = ((row + 1) % -M) % M
        output[row, features] = x[shifted_row, features]


@intent.kernel
def grouped_query_head_add(
    query: I.In[I.f16, ("HQ", "N")],
    key: I.In[I.f16, ("HK", "N")],
    output: I.Out[I.f16, ("HQ", "N")],
):
    HQ, N = query.shape
    tokens = I.domain(0, N)
    for query_head in I.parallel(I.domain(0, HQ)):
        key_head = (
            (query_head - GQA_QUERY_HEADS) // GQA_HEAD_GROUP
            + GQA_KEY_HEADS
        )
        output[query_head, tokens] = (
            query[query_head, tokens] + key[key_head, tokens]
        )


@intent.kernel
def scalar_table_lookup(
    labels: I.In[I.i32, ("M",)],
    table: I.In[I.f32, ("C", "D")],
    output: I.Out[I.f32, ("M", "D")],
):
    M = labels.shape[0]
    D = table.shape[1]
    features = I.domain(0, D)
    for row in I.parallel(I.domain(0, M)):
        label = labels[row]
        I.assume_in_bounds(label, table, axis=0)
        output[row, features] = table[label, features]
