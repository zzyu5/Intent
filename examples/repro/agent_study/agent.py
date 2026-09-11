from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

import tomli
import triton
import triton.language

from intent.tools.manual import snapshot


def _toml(value) -> str:
    if isinstance(value, dict):
        return "{" + ", ".join(f"{json.dumps(key)} = {_toml(item)}" for key, item in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_toml(item) for item in value) + "]"
    return json.dumps(value)


def materialize_language(project: Path, triton_ref: Path, directory: Path, language: str) -> list[str]:
    directory.mkdir(parents=True)
    if language == "intent":
        corpus = snapshot(project)
        (directory / "manual.json").write_text(json.dumps(corpus, ensure_ascii=False))
        return sorted({entry["source"] for entry in corpus["documents"].values()})
    language_path = Path(triton.language.__file__).parent
    sources = []
    for name in ("core.py", "standard.py", "math.py", "extra/cuda/libdevice.py"):
        destination = directory / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(language_path / name, destination)
        sources.append(f"triton-{triton.__version__}/language/{name}")
    path = triton_ref / "docs/python-api/triton-semantics.rst"
    shutil.copyfile(path, directory / path.name)
    sources.append("ref/triton/docs/python-api/triton-semantics.rst")
    example = Path(__file__).with_name("materials") / "triton/vector_add.py"
    shutil.copyfile(example, directory / "vector_add.py")
    return sources + [str(example.relative_to(project))]


def command(directory: Path, suite: dict, response: Path, schema: Path, executable: Path,
            state_root: Path, language: str) -> list[str]:
    config = tomli.loads((state_root / "config.toml").read_text())
    if (config["model"], config["model_reasoning_effort"]) != (suite["model"], suite["reasoning_effort"]):
        raise ValueError("dedicated configuration must retain the agreed model and reasoning effort")
    provider = config["model_provider"]
    if config["model_providers"][provider]["env_key"] != "INTENT_STUDY_API_KEY":
        raise ValueError("dedicated provider must use INTENT_STUDY_API_KEY")
    values = {
        "model_reasoning_effort": suite["reasoning_effort"],
        "model_provider": provider,
        f"model_providers.{provider}": config["model_providers"][provider],
        "approval_policy": "never",
        "default_permissions": "study",
        "permissions.study.filesystem": {":root": "deny", ":minimal": "read",
                                         ":workspace_roots": {".": "write", "materials": "read", "TASK.md": "read"},
                                         str(executable): "read", str(state_root / "codex/tmp/arg0"): "read"},
        "permissions.study.network.enabled": False,
        "project_doc_max_bytes": 0,
        "developer_instructions": Path(__file__).with_name("instructions.md").read_text(),
        "memories.generate_memories": False, "memories.use_memories": False,
        "agents.enabled": False, "web_search": "disabled",
        "shell_environment_policy.inherit": "none",
        "shell_environment_policy.set": {"PATH": "/usr/local/bin:/usr/bin:/bin", "PYTHONNOUSERSITE": "1"},
        "allow_login_shell": False, "features.skip_host_skill_discovery": True,
    }
    if language == "intent":
        project = Path(__file__).resolve().parents[3]
        values["mcp_servers.intent_manual"] = {
            "command": "/usr/bin/env",
            "args": ["-i", "PATH=/usr/local/bin:/usr/bin:/bin", f"PYTHONPATH={project / 'python'}",
                     sys.executable, "-B", "-m", "intent.tools.manual", "--corpus", str(directory / "materials/manual.json")],
            "required": True, "enabled_tools": ["search", "api", "read"],
            "default_tools_approval_mode": "approve",
        }
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


def execute(directory: Path, suite: dict, prompt: str, *, executable: Path,
            state_root: Path, language: str, stop: threading.Event) -> dict:
    response, schema = directory / "response.json", directory / "response-schema.json"
    schema.write_text(json.dumps({"type": "object", "additionalProperties": False,
                                  "properties": {"action": {"type": "string", "enum": ["submit"]},
                                                 "reason": {"type": "string"}},
                                  "required": ["action", "reason"]}))
    secret_path = state_root / "provider.key"
    if secret_path.stat().st_mode & 0o077:
        raise PermissionError("dedicated provider.key must only be accessible to its owner")
    key = secret_path.read_text().strip()
    if not key:
        raise ValueError("dedicated provider.key is empty")
    environment = {name: os.environ[name] for name in ("PATH", "HOME", "USER", "LANG", "TMPDIR") if name in os.environ}
    environment.update(CODEX_HOME=str(state_root / "codex"), INTENT_STUDY_API_KEY=key)
    started = time.monotonic()
    process = subprocess.Popen(command(directory, suite, response, schema, executable, state_root, language),
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               cwd=directory, env=environment, start_new_session=True)
    threads, errors, mcp_calls, stderr = [], [], [], []

    def collect_stdout():
        for line in process.stdout:
            if not line.startswith("{"):
                continue
            event = json.loads(line.replace(key, "<redacted>"))
            if event["type"] == "thread.started":
                threads.append(event["thread_id"])
            elif event["type"] in {"error", "turn.failed"}:
                errors.append(event)
            elif event["type"] == "item.completed" and event["item"]["type"] == "mcp_tool_call":
                item = event["item"]
                call = {k: item[k] for k in ("server", "tool", "arguments", "status", "error") if k in item}
                mcp_calls.append(call)
                print(json.dumps({"task_directory": str(directory), "manual_call": call}), flush=True)

    def collect_stderr():
        for line in process.stderr:
            stderr.append(line.replace(key, "<redacted>"))

    readers = [threading.Thread(target=collect_stdout), threading.Thread(target=collect_stderr)]
    for reader in readers:
        reader.start()
    process.stdin.write(prompt)
    process.stdin.close()
    timed_out = False
    while process.poll() is None:
        timed_out = time.monotonic() - started >= suite["agent_seconds"]
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
    result = {"model": suite["model"], "reasoning_effort": suite["reasoning_effort"],
              "threads": threads, "exit_code": process.returncode, "manual_calls": mcp_calls}
    if timed_out or stop.is_set() or process.returncode or not response.exists():
        result.update(action="unavailable", status="agent_timeout" if timed_out else "agent_environment_failure",
                      errors=errors, error="".join(stderr)[-6000:] or "Agent stopped without a submission")
    else:
        result.update(json.loads(response.read_text().replace(key, "<redacted>")))
    return result
