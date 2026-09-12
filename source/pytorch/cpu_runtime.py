import torch
import torch.nn.functional as F


def gelu(x):
    return F.gelu(x, approximate="tanh")


def relu(x):
    return torch.maximum(x, x.new_zeros(()))


def addcmul(x, scale, bias):
    return bias.unsqueeze(-1) + x * scale.unsqueeze(-1)


def logsumexp(x):
    return torch.logsumexp(x, dim=-1)


def softmax(x):
    values = x.float()
    numerator = torch.exp(values - values.amax(dim=-1, keepdim=True))
    return (numerator / numerator.sum(dim=-1, keepdim=True)).to(x.dtype)


def layer_norm(x, weight, bias, inverse_features, epsilon):
    values = x.float()
    mean = values.sum(dim=-1, keepdim=True) * inverse_features
    centered = values - mean
    variance = (centered * centered).sum(dim=-1, keepdim=True) * inverse_features
    return (centered * torch.rsqrt(variance + epsilon) * weight.float() + bias.float()).to(x.dtype)


def rms_norm(x, inverse_features, epsilon):
    mean_square = (x * x).sum(dim=-1, keepdim=True) * inverse_features
    return x * torch.rsqrt(mean_square + epsilon)


def weighted_rms_norm(x, weight, inverse_features, epsilon):
    values = x.float()
    return (rms_norm(values, inverse_features, epsilon) * weight.float()).to(x.dtype)


def softmax_backward(probabilities, upstream):
    projection = (probabilities * upstream).sum(dim=-1, keepdim=True)
    return probabilities * (upstream - projection)


def transpose(x):
    return x.T.contiguous()


def matmul(a, b):
    return torch.matmul(a.float(), b.float()).to(a.dtype)
