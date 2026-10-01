"""Pinned Nemotron OpenHands filter and conversion contracts."""

from __future__ import annotations

from collections import Counter
import json
import re
from typing import Any

from pgsft.data.canonical import CanonicalTrace, Message


NEMOTRON_REPO = "nvidia/Nemotron-SFT-SWE-v3"
NEMOTRON_REVISION = "3f73de64c1fe928a8f538fe45ccc10c228cc4c6a"
FILTER_VERSION = "nemotron-openhands-v1"
THINK_ACK = "Your thought has been logged."
ALLOWED_TOOLS = frozenset({"execute_bash", "str_replace_editor", "finish", "think"})
ISSUE_RE = re.compile(r"<issue_description>\s*(.*?)\s*</issue_description>", re.S)
WORKSPACE_RE = re.compile(r"/workspace(/[A-Za-z0-9_.\-]+)?")
TEST_CMD_RE = re.compile(
    r"\b(pytest|py\.test|tox|nosetests|python\S* -m unittest|go test|cargo test|npm test)\b"
)
TEST_FAIL_RE = re.compile(r"FAILED|ERROR")

SYSTEM_PROMPT = (
    "You are a software engineer AI fixing bugs at /testbed.\n"
    "\n"
    "OUTPUT FORMAT:\n"
    "Think first, then make exactly ONE tool call:\n"
    "\n"
    "<tool_call>\n"
    '{"name": "execute_bash", "arguments": {"command": "the shell command to run"}}\n'
    "</tool_call>\n"
    "\n"
    "Available tools:\n"
    "1. execute_bash — run a shell command in /testbed (explore, read files, run tests).\n"
    '   {"name": "execute_bash", "arguments": {"command": "<command>"}}\n'
    "2. str_replace_editor — custom editing tool for viewing, creating and editing files.\n"
    "   State is persistent across command calls. Commands: view, create, str_replace,\n"
    "   insert, undo_edit. Parameters: command, path, file_text,\n"
    "   old_str, new_str, insert_line, view_range.\n"
    '   {"name": "str_replace_editor", "arguments": {"command": "str_replace", '
    '"path": "/testbed/file.py", "old_str": "...", "new_str": "..."}}\n'
    "3. finish — call when the fix is complete and verified. Your workspace changes\n"
    "   (git diff vs the base commit) are collected automatically. Submitting with\n"
    "   NO code changes is rejected: you will be asked to continue and must actually\n"
    "   edit files in /testbed first.\n"
    '   {"name": "finish", "arguments": {"final_output": "<summary of your fix>"}}\n'
    "\n"
    "RULES:\n"
    "- Call exactly ONE <tool_call> block.\n"
    "- If you output multiple <tool_call> blocks, only the LAST one is executed.\n"
    '- The <tool_call> body is a single JSON object with "name" and "arguments".\n'
    "- Do NOT wrap the JSON in markdown code fences.\n"
    "- A text-only response (no <tool_call> block) is not accepted, and you will be asked to continue.\n"
    "\n"
    "WORKFLOW:\n"
    "1. Explore: find and read relevant files\n"
    "2. Reproduce: create and run a test script\n"
    "3. Fix: edit source code (NOT test files, NOT config files)\n"
    "4. Verify: run your test\n"
    "5. Finish: call the finish tool\n"
)


class SourceRecordError(ValueError):
    pass


def _arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    value = message.get("tool_calls") or []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise SourceRecordError("tool_calls is invalid JSON") from error
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise SourceRecordError("tool_calls must be a list of mappings")
    return value


def _normalize_path(text: str) -> str:
    return WORKSPACE_RE.sub("/testbed", text)


def filter_nemotron(row: dict[str, Any]) -> str | None:
    """Return the first paper filter rejection reason, or ``None``."""
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return "not_openhands"
    first = messages[0]
    if (
        not isinstance(first, dict)
        or first.get("role") != "system"
        or "OpenHands agent" not in str(first.get("content") or "")
    ):
        return "not_openhands"
    assistants = [message for message in messages if message.get("role") == "assistant"]
    if not assistants:
        return "sparse_reasoning"
    dense = sum(bool(str(message.get("reasoning_content") or "").strip()) for message in assistants)
    if dense / len(assistants) < 0.7:
        return "sparse_reasoning"
    try:
        calls = [call for message in assistants for call in _tool_calls(message)]
    except SourceRecordError:
        return "invalid_tool_calls"
    names = {(call.get("function") or {}).get("name") for call in calls}
    if "finish" not in names:
        return "no_finish"
    if not names <= ALLOWED_TOOLS:
        return "unknown_tool"
    seen_test = False
    pending_test = False
    last_observation: str | None = None
    for message in messages:
        if message.get("role") == "assistant":
            pending_test = False
            for call in _tool_calls(message):
                function = call.get("function") or {}
                if function.get("name") == "execute_bash":
                    command = str(_arguments(function.get("arguments")).get("command", ""))
                    if TEST_CMD_RE.search(command):
                        seen_test = True
                        pending_test = True
        elif message.get("role") == "tool" and pending_test:
            last_observation = str(message.get("content") or "")
            pending_test = False
    if not seen_test or (last_observation is not None and TEST_FAIL_RE.search(last_observation)):
        return "tail_test_failed"
    return None


def _tool_block(name: str, arguments: dict[str, Any]) -> str:
    return "<tool_call>\n" + json.dumps(
        {"name": name, "arguments": arguments}, ensure_ascii=False, sort_keys=True
    ) + "\n</tool_call>"


def canonicalize_nemotron(row: dict[str, Any], *, source_index: int) -> CanonicalTrace:
    """Convert one already-filtered OpenHands row to the public trace schema."""
    rejection = filter_nemotron(row)
    if rejection is not None:
        raise SourceRecordError(rejection)
    trace_id = row.get("uuid")
    if not isinstance(trace_id, str) or not trace_id.strip():
        raise SourceRecordError("missing_uuid")
    output: list[Message] = [Message("system", SYSTEM_PROMPT)]
    first_user = False
    pending_responses: list[str] = []
    buffered_reasoning: list[str] = []
    finished = False
    for raw_index, raw in enumerate(row["messages"]):
        if finished:
            break
        role = raw.get("role")
        if role == "system":
            continue
        if role == "tool":
            if not pending_responses:
                raise SourceRecordError(f"orphan_tool:{raw_index}")
            expected = pending_responses.pop(0)
            content = str(raw.get("content") or "")
            if expected == "think":
                if content != THINK_ACK:
                    raise SourceRecordError(f"think_ack_mismatch:{raw_index}")
                continue
            output.append(Message("tool", _normalize_path(content)))
            continue
        if pending_responses:
            raise SourceRecordError(f"missing_tool_response:{raw_index}")
        if role == "user":
            if buffered_reasoning:
                raise SourceRecordError(f"orphan_reasoning:{raw_index}")
            content = str(raw.get("content") or "")
            if not first_user:
                first_user = True
                match = ISSUE_RE.search(content)
                issue = match.group(1).strip() if match else content.strip()
                content = (
                    f"<problem>\n{_normalize_path(issue)}\n</problem>\n\n"
                    "Work in /testbed. Start by exploring the codebase."
                )
            output.append(Message("user", _normalize_path(content)))
            continue
        if role != "assistant":
            raise SourceRecordError(f"unexpected_role:{raw_index}")
        reasoning = str(raw.get("reasoning_content") or "").strip()
        reasoning_parts = [_normalize_path(reasoning)] if reasoning else []
        action_parts = []
        natural = _normalize_path(str(raw.get("content") or "").strip())
        if natural:
            action_parts.append(natural)
        saw_think_only = False
        for call in _tool_calls(raw):
            function = call.get("function") or {}
            name = function.get("name")
            arguments = _arguments(function.get("arguments"))
            if name == "think":
                thought = str(arguments.get("thought") or "").strip()
                if thought:
                    reasoning_parts.append(_normalize_path(thought))
                pending_responses.append("think")
                saw_think_only = True
                continue
            if name == "execute_bash":
                arguments = {"command": _normalize_path(str(arguments.get("command", "")))}
            elif name == "str_replace_editor":
                arguments = {
                    key: _normalize_path(value) if isinstance(value, str) else value
                    for key, value in arguments.items()
                }
            elif name == "finish":
                arguments = {
                    "final_output": str(
                        arguments.get("message") or arguments.get("final_output") or ""
                    )
                }
                finished = True
            else:
                raise SourceRecordError(f"unknown_tool:{name}")
            action_parts.append(_tool_block(str(name), arguments))
            if name != "finish":
                pending_responses.append("executable")
        if saw_think_only and not action_parts:
            buffered_reasoning.extend(reasoning_parts)
            continue
        if buffered_reasoning:
            reasoning_parts = [*buffered_reasoning, *reasoning_parts]
            buffered_reasoning.clear()
        if not action_parts:
            raise SourceRecordError(f"empty_action:{raw_index}")
        output.append(
            Message(
                "assistant",
                "\n\n".join(action_parts),
                "\n\n".join(reasoning_parts),
            )
        )
    if pending_responses or buffered_reasoning or not finished:
        raise SourceRecordError("incomplete_trajectory")
    return CanonicalTrace(
        id=trace_id,
        messages=tuple(output),
        source={"dataset": NEMOTRON_REPO, "row_index": source_index},
    )
