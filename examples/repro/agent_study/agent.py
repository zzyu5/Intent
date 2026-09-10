from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time

import tomli
import triton
import triton.language


def _toml(value) -> str:
    if isinstance(value, dict):
        return "{" + ", ".join(f"{json.dumps(key)} = {_toml(item)}" for key, item in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_toml(item) for item in value) + "]"
    return json.dumps(value)


def materialize_language(project: Path, triton_ref: Path, directory: Path, language: str) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    sources = []
    if language == "intent":
        paths = [project / "doc/dsl" / name for name in ("README.md", "core.md", "types-numerics-and-effects.md")]
        paths += [project / "doc/programming-model" / name for name in ("README.md", "logical-program.md", "kernel-and-host.md")]
        for path in paths:
            relative = path.relative_to(project / "doc")
            output = directory / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, output)
            sources.append(str(path.relative_to(project)))
    else:
        # Installed APIs match the executing Triton, not a potentially newer ref.
        language_path = Path(triton.language.__file__).parent
        for name in ("core.py", "standard.py", "math.py", "extra/cuda/libdevice.py"):
            path = language_path / name
            destination = directory / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            sources.append(f"triton-{triton.__version__}/language/{name}")
        path = triton_ref / "docs/python-api/triton-semantics.rst"
        shutil.copyfile(path, directory / path.name)
        sources.append("ref/triton/docs/python-api/triton-semantics.rst")
    example = Path(__file__).with_name("materials") / language / "vector_add.py"
    shutil.copyfile(example, directory / "vector_add.py")
    sources.append(str(example.relative_to(project)))
    return sources


def command(directory: Path, suite: dict, response: Path, schema: Path, executable: Path) -> list[str]:
    codex_dir = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    config = tomli.loads((codex_dir / "config.toml").read_text())
    provider = config.get("model_provider", "openai")
    values = {
        "model_reasoning_effort": suite["reasoning_effort"],
        "model_provider": provider,
        "approval_policy": "never",
        "default_permissions": "study",
        "permissions.study.filesystem": {":root": "deny", ":minimal": "read",
                                         ":workspace_roots": {".": "write", "materials": "read", "TASK.md": "read"},
                                         str(executable): "read", str(codex_dir / "tmp/arg0"): "read"},
        "permissions.study.network.enabled": False,
        "project_doc_max_bytes": 0,
        "developer_instructions": Path(__file__).with_name("instructions.md").read_text(),
        "memories.generate_memories": False,
        "memories.use_memories": False,
        "agents.enabled": False,
        "web_search": "disabled",
        "shell_environment_policy.inherit": "none",
        "shell_environment_policy.set": {"PATH": "/usr/local/bin:/usr/bin:/bin", "PYTHONNOUSERSITE": "1"},
        "allow_login_shell": False,
        "features.skip_host_skill_discovery": True,
    }
    if provider in config.get("model_providers", {}):
        values[f"model_providers.{provider}"] = config["model_providers"][provider]
    result = [str(executable), "exec", "--ignore-user-config", "--ignore-rules", "--strict-config",
              "--ephemeral", "--skip-git-repo-check", "--json", "--color", "never",
              "--model", suite["model"], "--cd", str(directory),
              "--output-last-message", str(response), "--output-schema", str(schema)]
    for feature in ("multi_agent", "multi_agent_v2", "memories", "hooks", "plugins", "apps",
                    "browser_use", "computer_use", "image_generation", "shell_snapshot", "skill_search",
                    "unbounded_connection_retries", "view_image"):
        result += ["--disable", feature]
    for key, value in values.items():
        result += ["-c", f"{key}={_toml(value)}"]
    return result + ["-"]


def execute(directory: Path, suite: dict, prompt: str, *, remaining_seconds: float,
            executable: Path, stop: threading.Event) -> dict:
    response = directory / "response.json"
    schema = directory / "response-schema.json"
    schema.write_text(json.dumps({"type": "object", "additionalProperties": False,
                                  "properties": {"action": {"type": "string", "enum": ["submit", "stop"]},
                                                 "reason": {"type": "string"}},
                                  "required": ["action", "reason"]}))
    if response.exists():
        response.unlink()
    started = time.monotonic()
    process = subprocess.Popen(command(directory, suite, response, schema, executable), stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               cwd=directory, start_new_session=True)
    stdout_lines, stderr_lines, tool_activity, environment_errors = [], [], [], []

    def collect_stdout():
        for line in process.stdout:
            stdout_lines.append(line)
            if line.startswith("{"):
                event = json.loads(line)
                if event["type"] in {"thread.started", "turn.completed", "turn.failed", "error"}:
                    print(json.dumps({"agent_directory": str(directory), **event}), flush=True)
                elif event["type"] == "item.completed":
                    item = event["item"]
                    if item["type"] == "command_execution":
                        tool_activity.append({key: item[key] for key in ("command", "exit_code", "status") if key in item})
                        if item.get("exit_code", 0):
                            output = item.get("aggregated_output", "")
                            print(json.dumps({"agent_directory": str(directory), "tool_failure": tool_activity[-1],
                                              "error_excerpt": output[:1000]}), flush=True)
                            if "bwrap:" in output:
                                environment_errors.append(output)
                                stop.set()

    def collect_stderr():
        for line in process.stderr:
            stderr_lines.append(line)
            if "ERROR" in line:
                print(f"Agent tool error ({directory.parent.parent.name}/{directory.parent.parent.parent.name}): {line.strip()}", flush=True)

    readers = [threading.Thread(target=collect_stdout), threading.Thread(target=collect_stderr)]
    for reader in readers:
        reader.start()
    process.stdin.write(prompt)
    process.stdin.close()
    timed_out = False
    while process.poll() is None:
        timed_out = time.monotonic() - started >= remaining_seconds
        if timed_out or stop.is_set():
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            break
        stop.wait(1)
    for reader in readers:
        reader.join()
    stdout, stderr = "".join(stdout_lines), "".join(stderr_lines)
    usage, threads, errors = [], [], []
    for line in stdout.splitlines():
        if not line.startswith("{"):
            continue
        event = json.loads(line)
        if event["type"] == "thread.started":
            threads.append(event["thread_id"])
        elif event["type"] == "turn.completed":
            usage.append(event["usage"])
        elif event["type"] in {"error", "turn.failed"}:
            errors.append(event)
    result = {"model": suite["model"], "reasoning_effort": suite["reasoning_effort"],
              "seconds": time.monotonic() - started, "usage": usage, "threads": threads,
              "exit_code": process.returncode, "tool_activity": tool_activity}
    if timed_out or stop.is_set() or process.returncode or not response.exists():
        result.update(action="unavailable", status="agent_timeout" if timed_out else "agent_environment_failure",
                      errors=errors, error="\n".join(environment_errors) or stderr[-6000:] or "Agent execution stopped before a submission")
    else:
        result.update(json.loads(response.read_text()))
    return result
