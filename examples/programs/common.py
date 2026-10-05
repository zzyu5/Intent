"""Ordinary example invocation through the public artifact API."""

from dataclasses import dataclass, field

import intent
import torch


@dataclass
class ExampleContext:
    target: object
    device: torch.device
    compiler: str | None = None
    prepared: bool = False
    artifacts: list = field(default_factory=list)

    def compile(self, definition, *, constexprs=None):
        artifact = intent.compile(definition, target=self.target,
                                  compiler=self.compiler, constexprs=constexprs)
        self.artifacts.append(artifact)
        return artifact

    def call(self, artifact, *arguments):
        if not self.prepared:
            return artifact.run(*arguments)
        invocation = artifact.prepare(*arguments)
        invocation.launch()
        return invocation.result()


def describe_outputs(value, label="result"):
    if isinstance(value, torch.Tensor):
        print(f"{label}: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            describe_outputs(item, f"{label}[{index}]")
    elif isinstance(value, dict):
        for name, item in value.items():
            describe_outputs(item, f"{label}.{name}")
    else:
        print(f"{label}: {value}")
