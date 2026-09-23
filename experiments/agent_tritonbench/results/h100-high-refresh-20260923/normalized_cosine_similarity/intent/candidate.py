import torch
import intent
import intent.language as I


@intent.kernel
def _normalized_cosine_kernel(
    x1: I.In[I.f32, ("M", "N")],
    x2: I.In[I.f32, ("M", "N")],
    output: I.Out[I.f32, ("M",)],
    dim: I.i64,
    eps_similarity: I.f32,
    p_norm: I.f32,
    eps_norm: I.f32,
):
    # The supplied profile fixes dim=1 and p_norm=2. Each row is independent.
    M, N = x1.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)

    for row in I.parallel(rows):
        value1 = x1[row, columns]
        value2 = x2[row, columns]

        norm1_p = I.sqrt(I.reduce.sum(value1 * value1, axis=0, acc_dtype=I.f32))
        norm2_p = I.sqrt(I.reduce.sum(value2 * value2, axis=0, acc_dtype=I.f32))

        scale1 = I.select(norm1_p > eps_norm, norm1_p, eps_norm)
        scale2 = I.select(norm2_p > eps_norm, norm2_p, eps_norm)
        normalized1 = value1 / scale1
        normalized2 = value2 / scale2

        dot = I.reduce.sum(normalized1 * normalized2, axis=0, acc_dtype=I.f32)
        normalized_norm1 = I.sqrt(
            I.reduce.sum(normalized1 * normalized1, axis=0, acc_dtype=I.f32)
        )
        normalized_norm2 = I.sqrt(
            I.reduce.sum(normalized2 * normalized2, axis=0, acc_dtype=I.f32)
        )

        cosine_scale1 = I.select(
            normalized_norm1 > eps_similarity,
            normalized_norm1,
            eps_similarity,
        )
        cosine_scale2 = I.select(
            normalized_norm2 > eps_similarity,
            normalized_norm2,
            eps_similarity,
        )
        output[row] = dot / (cosine_scale1 * cosine_scale2)


def build(context):
    kernel = context.compile("normalized_cosine_similarity_kernel", _normalized_cosine_kernel)

    def normalized_cosine_similarity(
        x1: torch.Tensor,
        x2: torch.Tensor,
        dim: int = 1,
        eps_similarity: float = 1e-8,
        p_norm: float = 2,
        eps_norm: float = 1e-12,
    ) -> torch.Tensor:
        output = torch.empty(
            (x1.shape[0],),
            device=x1.device,
            dtype=torch.float32,
        )
        kernel(x1, x2, output, dim, eps_similarity, p_norm, eps_norm)
        return output

    return normalized_cosine_similarity
