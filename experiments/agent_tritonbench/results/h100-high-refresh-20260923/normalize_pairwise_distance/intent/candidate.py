import torch
import intent
import intent.language as I


@intent.kernel
def pairwise_distance_kernel(
    x1: I.In[I.f32, ("B", "M", "D")],
    x2: I.In[I.f32, ("B", "M", "D")],
    distance: I.Out[I.f32, ("B", "M")],
    eps_distance: I.f32,
):
    batch = I.domain(0, x1.shape[0])
    items = I.domain(0, x1.shape[1])
    features = I.domain(0, x1.shape[2])

    delta = x1[batch, items, features] - x2[batch, items, features]
    delta = delta + eps_distance
    squared = delta * delta
    distance_squared = I.reduce.sum(squared, axis=2, acc_dtype=I.f32)
    distance[batch, items] = I.sqrt(distance_squared)


@intent.kernel
def normalize_distance_kernel(
    distance: I.In[I.f32, ("B", "M")],
    output: I.Out[I.f32, ("B", "M")],
    eps_norm: I.f32,
):
    batch = I.domain(0, distance.shape[0])
    items = I.domain(0, distance.shape[1])

    values = distance[batch, items]
    norm_squared = I.reduce.sum(values * values, axis=1, acc_dtype=I.f32)
    norm = I.sqrt(norm_squared)
    denominator = I.maximum(norm, eps_norm)
    denominator = I.reshape(denominator, (distance.shape[0], 1))
    output[batch, items] = I.fdiv(values, denominator)


def build(context):
    distance_artifact = context.compile("pairwise_distance", pairwise_distance_kernel)
    normalize_artifact = context.compile("normalize_distance", normalize_distance_kernel)

    def normalize_pairwise_distance(
        x1,
        x2,
        p_distance=2.0,
        eps_distance=1e-6,
        keepdim=False,
        p_norm=2,
        dim_norm=1,
        eps_norm=1e-12,
    ):
        distance = torch.empty(
            (x1.shape[0], x1.shape[1]),
            device=x1.device,
            dtype=x1.dtype,
        )
        output = torch.empty_like(distance)

        distance_artifact(x1, x2, distance, eps_distance)
        normalize_artifact(distance, output, eps_norm)
        return output

    return normalize_pairwise_distance
