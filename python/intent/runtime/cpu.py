"""CPU invocation legality shared by Mojo and Weft."""

from .native import AliasCheck, ViewFacts


def check_alias(check: AliasCheck, left: ViewFacts, right: ViewFacts) -> None:
    if check.writable_overlap(left, right):
        raise NotImplementedError("overlapping writable CPU views are not implemented")
    if check.noalias_violation(left, right):
        raise ValueError("CPU invocation violates an author noalias constraint")
    if check.allocation_violation(left, right):
        raise ValueError("CPU invocation violates an author allocation-alias constraint")
