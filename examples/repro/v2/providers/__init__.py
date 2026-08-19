from __future__ import annotations

from importlib import import_module


def load_cases(provider: str):
    return import_module(f"repro.v2.providers.{provider}").CASES

