import torch
import torch.nn.functional as F


def gelu(x):
    return F.gelu(x, approximate="tanh")


def relu(x):
    return torch.maximum(x, x.new_zeros(()))


def swiglu(gate, up):
    values = gate.float()
    sigmoid = 1.0 / (1.0 + torch.exp(-values))
    sigmoid = torch.where(sigmoid < torch.finfo(torch.float32).tiny, sigmoid * 0.0, sigmoid)
    return (values * sigmoid).to(gate.dtype) * up


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


def index_select(source, indices):
    return torch.index_select(source, 0, indices)


def matmul(a, b):
    return torch.matmul(a.float(), b.float()).to(a.dtype)


def conv1d_same(x, weight):
    width = weight.numel()
    patches = F.pad(x.float(), (width // 2, width // 2)).unfold(-1, width, 1)
    return (patches * weight.float()).sum(dim=-1).to(x.dtype)


def adamw(gradient, parameter, first_moment, second_moment, learning_rate,
          beta1, beta2, bias_correction1, bias_correction2, epsilon, weight_decay):
    first = beta1 * first_moment + (1.0 - beta1) * gradient
    second = beta2 * second_moment + (1.0 - beta2) * gradient * gradient
    denominator = torch.sqrt(second / bias_correction2) + epsilon
    update = (first / bias_correction1) / denominator + weight_decay * parameter
    parameter.copy_(parameter - learning_rate * update)
    first_moment.copy_(first)
    second_moment.copy_(second)
    return parameter, first_moment, second_moment
