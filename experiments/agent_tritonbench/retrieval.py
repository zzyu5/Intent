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


class _StreamError(RuntimeError):
    def __init__(self, error):
        super().__init__(str(error))
        self.retryable = isinstance(error, dict) and str(error.get("code")) in {
            "408", "429", "500", "502", "503", "504", "INTERNAL_ERROR",
        }


async def _stream(url: str, body: dict, headers: dict, chunks: list[str],
                  stop: threading.Event, deadline: float, *, reasoning: list[str]) -> str | None:
    async def receive():
        finish = None
        async with httpx.AsyncClient(timeout=min(180, deadline - time.monotonic())) as client:
            async with client.stream("POST", url, json=body, headers=headers) as response:
                if response.is_error:
                    detail = (await response.aread()).decode(errors="replace")
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}: {detail[:4096]}",
                        request=response.request, response=response)
                async for line in response.aiter_lines():
                    line = line.strip()
                    if line == "data: [DONE]":
                        break
                    if not line.startswith("data:"):
                        continue
                    event = json.loads(line[5:])
                    if event.get("error"):
                        raise _StreamError(event["error"])
                    for part in event.get("choices", []):
                        chunks.append(part.get("delta", {}).get("content") or "")
                        reasoning.append(part.get("delta", {}).get("reasoning_content") or "")
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
        "Before drafting the program, query the API contracts you intend to use "
        "with api, read or search. Request documentation by returning a JSON object "
        'of the form {"queries": [{"tool": "api", "arguments": {"name": "I.full"}}]}. '
        "Available queries are api(name), read(id, section=null), and "
        "search(query, kind='all'); search kinds are all, api, concept and diagnostic. "
        "Use exact API names or document/section IDs returned by search. Multiple "
        "queries may be requested together. The caller returns only their public "
        "documentation; it cannot execute or validate a program. Continue the "
        "same task after reading those results.\n\n"
    ) + text[end:]
    begin = text.index("Write candidate.py to disk before finishing")
    return text[:begin] + (
        "Each response must be either a documentation queries JSON object as "
        "described above, or a final JSON object with one key, program, whose "
        "string value is the complete Python source for candidate.py. "
        "The caller saves the final program verbatim. "
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
    documents = [
        entry["id"] for entry in sorted(corpus["documents"].values(), key=lambda entry: entry["line"])
        if entry["source"] == "doc/dsl/authoring.md" and "#L" in entry["id"]
        and entry["title"] != "Host 编译与调用"
    ]
    documents.append(manual.api("context.compile")["rules"][0]["id"])
    bootstrap_count = len(documents)
    query = (directory / "TASK.md").read_text().split("Callable parameter signature", 1)[0]
    for hit in manual.search(query, kind="concept")["matches"]:
        if hit["source"] in {"doc/dsl/authoring.md", "doc/programming-model/kernel-and-host.md"}:
            continue
        if hit["id"] not in documents:
            documents.append(hit["id"])
        if len(documents) == bootstrap_count + 3:
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
              "delivery_continuations": 0, "request_retries": 0, "manual_queries": []}
    response_count = 0
    deadline = time.monotonic() + suite["agent_seconds"]
    while not stop.is_set() and time.monotonic() < deadline:
        body = {"model": suite["model"], "messages": messages,
                "reasoning_effort": suite["reasoning_effort"],
                "thinking": {"type": "enabled", "clear_thinking": False},
                "max_tokens": 32768, "response_format": {"type": "json_object"}, "stream": True}
        headers = {"Content-Type": "application/json", **provider.get("http_headers", {}),
                   "Authorization": "Bearer " + key}
        chunks, reasoning_chunks, finish, error = [], [], None, None
        retryable = False
        try:
            finish = asyncio.run(_stream(
                provider["base_url"].rstrip("/") + "/chat/completions",
                body, headers, chunks, stop, deadline,
                reasoning=reasoning_chunks))
        except (httpx.HTTPError, OSError, ValueError, RuntimeError, KeyError, TypeError) as failure:
            error = str(failure)
            retryable = (
                isinstance(failure, (httpx.TransportError, OSError))
                or isinstance(failure, httpx.HTTPStatusError)
                and failure.response.status_code in {408, 429, 500, 502, 503, 504}
                or isinstance(failure, _StreamError) and failure.retryable)
        content = "".join(chunks)
        reasoning = "".join(reasoning_chunks)
        turn = response_count
        response_count += 1
        (directory / f"response-{turn}.txt").write_text(content.replace(key, "<redacted>"))
        if reasoning:
            (directory / f"reasoning-{turn}.txt").write_text(reasoning.replace(key, "<redacted>"))
        result["finish_reason"] = finish
        previous = {"role": "assistant", "content": content}
        if reasoning:
            previous["reasoning_content"] = reasoning
        unfinished = content.count("<think>") > content.count("</think>")
        if finish == "stop" and not unfinished:
            source = content.strip()
            if source.startswith("<think>"):
                source = source.split("</think>", 1)[1].strip()
            try:
                reply = json.loads(source)
                if not isinstance(reply, dict):
                    raise ValueError("Response must be a JSON object")
                if set(reply) == {"queries"}:
                    calls = reply["queries"]
                    if not isinstance(calls, list) or not calls:
                        raise ValueError("queries must be a nonempty list")
                    responses = []
                    methods = {"api": manual.api, "read": manual.read, "search": manual.search}
                    for call in calls:
                        if not isinstance(call, dict) or set(call) != {"tool", "arguments"}:
                            raise ValueError("Each query must contain tool and arguments")
                        name, arguments = call["tool"], call["arguments"]
                        query = {"tool": name, "arguments": arguments}
                        try:
                            if not isinstance(name, str) or not isinstance(arguments, dict):
                                raise TypeError("Query tool must be a string and arguments an object")
                            if any(not isinstance(value, str) and not (key == "section" and value is None)
                                   for key, value in arguments.items()):
                                raise TypeError("Manual arguments must be strings; section may also be null")
                            response = methods[name](**arguments)
                            query["status"] = "returned"
                        except (KeyError, TypeError, ValueError) as failure:
                            response = {"error": str(failure)}
                            query["status"] = "error"
                        result["manual_queries"].append(query)
                        responses.append({**query, "response": response})
                    messages.extend((previous, {"role": "user", "content":
                        "Public manual query results:\n" + json.dumps(responses, ensure_ascii=False)}))
                    continue
                if set(reply) != {"program"}:
                    raise ValueError("Response must contain only queries or only program")
                program = reply["program"]
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
        if not content and not reasoning:
            if retryable:
                result.setdefault("interrupted_stream_errors", []).append(error.replace(key, "<redacted>"))
                result["request_retries"] += 1
                stop.wait(min(2 ** min(result["request_retries"], 4),
                              max(0, deadline - time.monotonic())))
                continue
            return {**result, "action": "unavailable",
                    "status": "agent_program_error" if finish == "length" else "agent_environment_failure",
                    "error": (error or "Empty incomplete response").replace(key, "<redacted>")}
        if error:
            result.setdefault("interrupted_stream_errors", []).append(error.replace(key, "<redacted>"))
        result.setdefault("interrupted_finish_reasons", []).append(finish)
        messages.extend((
            previous,
            {"role": "user", "content":
             "The response stream was interrupted. Continue the original task from the retained "
             "context and return a complete documentation query JSON or final program JSON. "
             "No compiler, execution or benchmark feedback is available."},
        ))
        result["delivery_continuations"] += 1
    return {**result, "action": "unavailable",
            "status": "agent_environment_failure" if stop.is_set() else "agent_timeout",
            "error": "Original generation deadline or stop request reached without a final source"}
