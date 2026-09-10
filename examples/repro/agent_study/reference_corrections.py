import torch


def sub_gelu(input, other, alpha=1, approximate="none", out=None):
    result = input - alpha * other
    if approximate == "tanh":
        result = 0.5 * result * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (result + 0.044715 * result ** 3)
        ))
    else:
        result = result * (1 + torch.erf(result / torch.sqrt(torch.tensor(2.0)))) * 0.5
    if out is not None:
        out.copy_(result)
        return out
    return result


CORRECTIONS = {"sub_gelu_standard_gelu": sub_gelu}
