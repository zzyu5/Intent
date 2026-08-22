from __future__ import annotations

from importlib import import_module

from ..model import ComparisonUnavailable


def implementation_gap(reason: str):
    def factory(_context):
        raise ComparisonUnavailable("intent_implementation_gap", reason)

    return factory


def load_cases(provider: str):
    return import_module(f"repro.v2.providers.{provider}").CASES
