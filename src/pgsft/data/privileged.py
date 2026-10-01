"""Fixed privileged-HINT transformation used by PG-SFT."""

from __future__ import annotations

from collections.abc import Callable

from pgsft.data.canonical import CanonicalTrace, Message


HINT_TEMPLATE = (
    "<HINT>\n"
    "A suggested action for the next step is:\n"
    "{action}\n"
    "Continue thinking about the next step.\n"
    "</HINT>"
)


def _assistant_content(message: Message) -> str:
    return message.content


EXTRACTORS: dict[str, Callable[[Message], str]] = {
    "assistant_content": _assistant_content,
}


def build_privileged_trace(
    trace: CanonicalTrace, *, extractor: str = "assistant_content"
) -> CanonicalTrace:
    """Append each exact assistant action to its immediate causal carrier."""
    try:
        extract = EXTRACTORS[extractor]
    except KeyError as error:
        raise ValueError(f"unsupported privileged extractor: {extractor}") from error
    messages = list(trace.messages)
    assistants = 0
    for index, message in enumerate(trace.messages):
        if message.role != "assistant":
            continue
        assistants += 1
        if index == 0 or messages[index - 1].role not in {"user", "tool"}:
            raise ValueError("assistant turn lacks preceding user/tool carrier")
        action = extract(message)
        if not isinstance(action, str) or not action.strip():
            raise ValueError("assistant action content must be non-empty")
        carrier = messages[index - 1]
        messages[index - 1] = Message(
            carrier.role,
            carrier.content + "\n\n" + HINT_TEMPLATE.format(action=action),
            carrier.reasoning_content,
        )
    if assistants == 0:
        raise ValueError("trace contains no assistant turn")
    return CanonicalTrace(
        id=trace.id,
        messages=tuple(messages),
        tools=trace.tools,
        source=trace.source,
    )
