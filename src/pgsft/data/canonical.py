"""Strict, JSON-serializable public trace schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence


Role = Literal["system", "user", "assistant", "tool"]
ROLES = frozenset({"system", "user", "assistant", "tool"})
_SECRET_FRAGMENTS = ("secret", "password", "passwd", "token", "api_key", "apikey")


def _text(value: Any, label: str, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "non-empty " if not allow_empty else ""
        raise ValueError(f"{label} must be a {qualifier}string")
    return value


def _validate_public_metadata(value: Any, *, path: str = "source") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            lowered = key.lower()
            if any(fragment in lowered for fragment in _SECRET_FRAGMENTS):
                raise ValueError(f"{path}.{key} may contain a secret")
            _validate_public_metadata(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_public_metadata(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and Path(value).is_absolute():
        raise ValueError(f"{path} contains an absolute local path")
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError(f"{path} contains a non-JSON value")


@dataclass(frozen=True)
class Message:
    role: Role
    content: str
    reasoning_content: str | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unsupported message role: {self.role!r}")
        _text(self.content, "message.content")
        if self.reasoning_content is not None:
            _text(self.reasoning_content, "message.reasoning_content")
        if self.role != "assistant" and self.reasoning_content is not None:
            raise ValueError("reasoning_content is only valid on assistant messages")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Message":
        allowed = {"role", "content", "reasoning_content"}
        unknown = set(value) - allowed
        missing = {"role", "content"} - set(value)
        if missing or unknown:
            raise ValueError(
                f"invalid message fields: missing={sorted(missing)} unknown={sorted(unknown)}"
            )
        return cls(
            role=value["role"],  # type: ignore[arg-type]
            content=value["content"],
            reasoning_content=value.get("reasoning_content"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.reasoning_content is not None:
            result["reasoning_content"] = self.reasoning_content
        return result


@dataclass(frozen=True)
class CanonicalTrace:
    id: str
    messages: tuple[Message, ...]
    tools: tuple[dict[str, Any], ...] = ()
    source: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _text(self.id, "trace.id", allow_empty=False)
        if not self.messages:
            raise ValueError("trace.messages must not be empty")
        if self.messages[0].role == "assistant":
            raise ValueError("assistant cannot be the first message")
        assistant_count = 0
        for index, message in enumerate(self.messages):
            if message.role != "assistant":
                continue
            assistant_count += 1
            if not message.content.strip():
                raise ValueError(f"assistant action content is empty at message {index}")
            if index == 0 or self.messages[index - 1].role not in {"user", "tool"}:
                raise ValueError(
                    f"assistant message {index} must immediately follow user or tool"
                )
        if assistant_count == 0:
            raise ValueError("trace must contain at least one assistant message")
        for tool in self.tools:
            if not isinstance(tool, dict):
                raise ValueError("trace.tools entries must be mappings")
            _validate_public_metadata(tool, path="tools")
        if self.source is not None:
            if not isinstance(self.source, dict):
                raise ValueError("trace.source must be a mapping or null")
            _validate_public_metadata(self.source)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CanonicalTrace":
        allowed = {"id", "messages", "tools", "source"}
        unknown = set(value) - allowed
        missing = {"id", "messages"} - set(value)
        if missing or unknown:
            raise ValueError(
                f"invalid trace fields: missing={sorted(missing)} unknown={sorted(unknown)}"
            )
        raw_messages = value["messages"]
        raw_tools = value.get("tools", [])
        if not isinstance(raw_messages, list) or not isinstance(raw_tools, list):
            raise ValueError("trace.messages and trace.tools must be lists")
        source = value.get("source")
        return cls(
            id=value["id"],
            messages=tuple(Message.from_mapping(message) for message in raw_messages),
            tools=tuple(dict(tool) for tool in raw_tools),
            source=None if source is None else dict(source),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "messages": [message.to_dict() for message in self.messages],
        }
        if self.tools:
            result["tools"] = [dict(tool) for tool in self.tools]
        if self.source is not None:
            result["source"] = self.source
        return result


def read_canonical_jsonl(path: Path) -> list[CanonicalTrace]:
    records: list[CanonicalTrace] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(raw, dict):
                raise ValueError(f"trace must be a mapping at {path}:{line_number}")
            trace = CanonicalTrace.from_mapping(raw)
            if trace.id in seen:
                raise ValueError(f"duplicate trace ID at {path}:{line_number}: {trace.id}")
            seen.add(trace.id)
            records.append(trace)
    return records


def write_canonical_jsonl(path: Path, records: Sequence[CanonicalTrace]) -> None:
    """Write deterministic compact JSONL without overwriting an existing output."""
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    seen: set[str] = set()
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            for trace in records:
                if trace.id in seen:
                    raise ValueError(f"duplicate trace ID: {trace.id}")
                seen.add(trace.id)
                handle.write(
                    json.dumps(
                        trace.to_dict(),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
