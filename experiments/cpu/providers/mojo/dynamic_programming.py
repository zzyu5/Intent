import torch

from kernels.dynamic_programming.smith_waterman import smith_waterman_score
from kernels.dynamic_programming.viterbi import viterbi_decode
from experiments._common.model import Tolerance
from .common import prepare_host_comparison


def smith_waterman(context):
    query = torch.randint(0, 20, (128, 128), dtype=torch.int32)
    reference = torch.randint(0, 20, (128, 128), dtype=torch.int32)
    return prepare_host_comparison(context, smith_waterman_score, (query, reference),
                                   "smith_waterman", Tolerance(atol=0.0))


def viterbi(context):
    emissions = torch.randn((64, 256, 64), dtype=torch.float32) * 0.1
    transitions = torch.randn((64, 64), dtype=torch.float32) * 0.1
    return prepare_host_comparison(context, viterbi_decode, (emissions, transitions), "viterbi",
                                   (Tolerance(atol=0.0), Tolerance(atol=2e-5)))


CASES = {"smith_waterman": smith_waterman, "viterbi": viterbi}
