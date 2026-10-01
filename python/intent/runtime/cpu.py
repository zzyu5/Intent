"""CPU invocation legality shared by Mojo and Weft."""

from .interface import AliasCheck, byte_spans_overlap
from .native import ViewFacts


def check_alias(check: AliasCheck, left: ViewFacts, right: ViewFacts) -> None:
    if check.writable and byte_spans_overlap((left.begin, left.end), (right.begin, right.end)):
        raise NotImplementedError("overlapping writable CPU views are not implemented")
    if check.noalias_violation((left.allocation, left.allocation_end), (right.allocation, right.allocation_end)):
        raise ValueError("CPU invocation violates an author noalias constraint")
    if check.allocation_violation(left.allocation, right.allocation):
        raise ValueError("CPU invocation violates an author allocation-alias constraint")
