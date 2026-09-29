from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
import threading
import time

import httpx
import tomli

from intent.tools.manual import Manual, _section_body
from .agent import instructions


async def _stream(url: str, body: dict, headers: dict, chunks: list[str],
                  stop: threading.Event, deadline: float) -> str | None:
    async def receive():
        finish = None
        async with httpx.AsyncClient(timeout=min(180, deadline - time.monotonic())) as client:
            async with client.stream("POST", url, json=body, headers=headers) as response:
                if response.is_error:
                    detail = (await response.aread()).decode(errors="replace")
                    raise RuntimeError(f"HTTP {response.status_code}: {detail[:4096]}")
                async for line in response.aiter_lines():
                    line = line.strip()
                    if line == "data: [DONE]":
                        break
                    if not line.startswith("data:"):
                        continue
                    event = json.loads(line[5:])
                    if event.get("error"):
                        raise RuntimeError(str(event["error"]))
                    for part in event.get("choices", []):
                        chunks.append(part.get("delta", {}).get("content") or "")
                        if part.get("finish_reason"):
                            finish = part["finish_reason"]
        return finish

    request = asyncio.create_task(receive())
    try:
        while not stop.is_set() and time.monotonic() < deadline:
            done, _ = await asyncio.wait({request}, timeout=min(0.25, deadline - time.monotonic()))
            if done:
                return request.result()
        raise TimeoutError("Original generation deadline or stop request reached")
    finally:
        if not request.done():
            request.cancel()
            try:
                await request
            except asyncio.CancelledError:
                pass


def generation_instructions() -> str:
    text = instructions("intent")
    begin = text.index("Read materials/api.txt")
    end = text.index("Intent expresses", begin)
    text = text[:begin] + (
        "Use the attached public API declarations and language contracts. "
        "No tools are available in this request. The material contains public "
        "syntax and semantics, not a task implementation.\n\n"
    ) + text[end:]
    begin = text.index("Write candidate.py to disk before finishing")
    return text[:begin] + (
        "Return a JSON object with one key, program, whose string value is the "
        "complete Python source for candidate.py. The caller saves it verbatim. "
        "Do not claim unmeasured correctness or speed."
    )


def execute(directory: Path, suite: dict, prompt: str, *,
            state_root: Path, stop: threading.Event) -> dict:
    config = tomli.loads((state_root / "config.toml").read_text())
    if (config["model"], config["model_reasoning_effort"]) != (
            suite["model"], suite["reasoning_effort"]):
        raise ValueError("dedicated configuration must retain the agreed model and reasoning effort")
    provider = config["model_providers"][config["model_provider"]]
    if provider["env_key"] != "INTENT_STUDY_API_KEY":
        raise ValueError("dedicated provider must use INTENT_STUDY_API_KEY")
    secret = state_root / "provider.key"
    if secret.stat().st_mode & 0o077:
        raise PermissionError("dedicated provider.key must only be accessible to its owner")
    key = secret.read_text().strip()
    if not key:
        raise ValueError("dedicated provider.key is empty")

    material = directory / "materials"
    corpus = json.loads((material / "manual.json").read_text())
    manual = Manual(corpus)
    documents = ["doc/dsl/authoring.md", manual.api("context.compile")["rules"][0]["id"]]
    query = (directory / "TASK.md").read_text().split("Callable parameter signature", 1)[0]
    for hit in manual.search(query, kind="concept")["matches"]:
        if hit["source"] in {"doc/dsl/authoring.md", "doc/programming-model/kernel-and-host.md"}:
            continue
        if hit["id"] not in documents:
            documents.append(hit["id"])
        if len(documents) == 5:
            break
    excerpts = []
    for identifier in documents:
        text = manual.read(identifier)["text"]
        excerpts.append("Source: " + identifier + "\n" +
                        (_section_body(text) if "#L" in identifier else text))
    public_text = (material / "api.txt").read_text() + "\n\n" + "\n\n".join(excerpts)
    messages = [
        {"role": "system", "content": generation_instructions()},
        {"role": "user", "content": prompt + "\n\nPUBLIC LANGUAGE CONTRACTS\n" + public_text},
    ]
    result = {"task_directory": str(directory), "model": suite["model"],
              "reasoning_effort": suite["reasoning_effort"], "generation_method": "public-manual-rag",
              "manual_revision": corpus["revision"], "documents": documents,
              "delivery_continuations": 0}
    deadline = time.monotonic() + suite["agent_seconds"]
    while not stop.is_set() and time.monotonic() < deadline:
        body = {"model": suite["model"], "messages": messages,
                "reasoning_effort": suite["reasoning_effort"],
                "thinking": {"type": "enabled", "clear_thinking": False},
                "max_tokens": 32768, "response_format": {"type": "json_object"}, "stream": True}
        headers = {"Content-Type": "application/json", **provider.get("http_headers", {}),
                   "Authorization": "Bearer " + key}
        chunks, finish, error = [], None, None
        try:
            finish = asyncio.run(_stream(
                provider["base_url"].rstrip("/") + "/chat/completions",
                body, headers, chunks, stop, deadline))
        except (httpx.HTTPError, OSError, ValueError, RuntimeError) as failure:
            error = str(failure)
        content = "".join(chunks)
        turn = result["delivery_continuations"]
        (directory / f"response-{turn}.txt").write_text(content.replace(key, "<redacted>"))
        result["finish_reason"] = finish
        unfinished = content.count("<think>") > content.count("</think>")
        if finish == "stop" and not unfinished:
            source = content.strip()
            if source.startswith("<think>"):
                source = source.split("</think>", 1)[1].strip()
            try:
                program = json.loads(source)["program"]
                if not isinstance(program, str):
                    raise ValueError("program must be a source string")
                tree = ast.parse(program)
                if not any(isinstance(node, ast.FunctionDef) and node.name == "build"
                           for node in tree.body):
                    raise ValueError("Final source does not define build(context)")
            except (ValueError, KeyError, SyntaxError, TypeError) as failure:
                return {**result, "action": "unavailable", "status": "agent_program_error",
                        "error": str(failure).replace(key, "<redacted>")}
            (directory / "candidate.py").write_text(program + "\n")
            return {**result, "action": "submit",
                    "reason": "Complete final source; no compiler or benchmark feedback"}
        if stop.is_set() or time.monotonic() >= deadline:
            break
        if not content:
            return {**result, "action": "unavailable",
                    "status": "agent_program_error" if finish == "length" else "agent_environment_failure",
                    "error": (error or "Empty incomplete response").replace(key, "<redacted>")}
        if error:
            result.setdefault("interrupted_stream_errors", []).append(error.replace(key, "<redacted>"))
        result.setdefault("interrupted_finish_reasons", []).append(finish)
        messages.extend((
            {"role": "assistant", "content": content},
            {"role": "user", "content":
             "The response stream was interrupted before your final source. Continue the original "
             "task from the retained context and return the complete program JSON. "
             "No compiler, execution or benchmark feedback is available."},
        ))
        result["delivery_continuations"] += 1
    return {**result, "action": "unavailable",
            "status": "agent_environment_failure" if stop.is_set() else "agent_timeout",
            "error": "Original generation deadline or stop request reached without a final source"}
