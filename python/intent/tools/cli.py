from __future__ import annotations

import argparse
import json
import sys

from .backends import BACKENDS
from .compilation import compile_request, doctor


def _assignments(parser, values: list[str]) -> dict:
    result = {}
    for item in values:
        name, separator, encoded = item.partition("=")
        if not separator or not name.isidentifier() or name in result:
            parser.error(f"expected a distinct NAME=JSON assignment, got {item!r}")
        try:
            result[name] = json.loads(encoded)
        except json.JSONDecodeError as error:
            parser.error(f"invalid JSON value for {name}: {error}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an Intent environment or compile an existing Python kernel")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("doctor", "Check selected backend dependencies and target facts"),
                            ("compile", "Compile a Python file:kernel or importable.module:kernel")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--target", choices=BACKENDS, required=True)
        command.add_argument("--target-option", action="append", default=[], metavar="NAME=JSON")
        command.add_argument("--compiler", help="Intent compiler override")
        command.add_argument("--json", action="store_true", help="Write a structured result")
    compile_parser = commands.choices["compile"]
    compile_parser.add_argument("program", metavar="PROGRAM:KERNEL")
    compile_parser.add_argument("--constexpr", action="append", default=[], metavar="NAME=JSON")
    compile_parser.add_argument("--tuning-config")
    compile_parser.add_argument("--materialize", action="store_true", help="Also create the callable; never launch the kernel")
    arguments = parser.parse_args()
    options = _assignments(parser, arguments.target_option)
    if arguments.command == "doctor":
        result = doctor(arguments.target, target_options=options, compiler=arguments.compiler)
    else:
        program, separator, kernel = arguments.program.rpartition(":")
        if not separator or not program:
            parser.error("program must be a Python file:kernel or importable.module:kernel")
        result = compile_request(program, kernel, arguments.target, target_options=options,
                                 constexprs=_assignments(parser, arguments.constexpr),
                                 compiler=arguments.compiler, tuning_config=arguments.tuning_config,
                                 materialize=arguments.materialize)
    if arguments.json:
        print(json.dumps(result, indent=2))
    elif arguments.command == "doctor":
        print(f"{arguments.target}: {result['status']}")
        for check in result["checks"]:
            print(f"  {check['name']}: {check['status']} — {check.get('message', check.get('detail'))}")
        print(result["scope"])
    else:
        print(f"{arguments.program}: {result['status']} for {arguments.target}")
        if "diagnostic" in result:
            print(f"Stage: {result['stage']}\n{result['diagnostic']['message']}", file=sys.stderr)
        for name, path in result.get("files", {}).items():
            print(f"  {name}: {path}")
        if result.get("program_stdout"):
            print(result["program_stdout"], file=sys.stderr, end="")
        print("The tool did not launch the selected kernel; module-level Python code executes normally.")
        print("Compilation does not establish numerical correctness.")
    if result["status"] in {"error", "unavailable"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
