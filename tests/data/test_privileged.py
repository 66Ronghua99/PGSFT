from __future__ import annotations

import pytest

from pgsft.data.canonical import CanonicalTrace, Message
from pgsft.data.privileged import HINT_TEMPLATE, build_privileged_trace


def _two_turn_trace() -> CanonicalTrace:
    return CanonicalTrace(
        id="pair",
        messages=(
            Message("user", "question"),
            Message("assistant", "first action", "first reason"),
            Message("tool", "observation"),
            Message("assistant", "second action", "second reason"),
        ),
    )


def _unsafe_trace(messages: tuple[Message, ...]) -> CanonicalTrace:
    trace = object.__new__(CanonicalTrace)
    object.__setattr__(trace, "id", "unsafe")
    object.__setattr__(trace, "messages", messages)
    object.__setattr__(trace, "tools", ())
    object.__setattr__(trace, "source", None)
    return trace


def test_hint_template_is_frozen_exactly() -> None:
    assert HINT_TEMPLATE == (
        "<HINT>\n"
        "A suggested action for the next step is:\n"
        "{action}\n"
        "Continue thinking about the next step.\n"
        "</HINT>"
    )


def test_one_exact_action_hint_is_appended_to_each_immediate_carrier() -> None:
    original = _two_turn_trace()
    privileged = build_privileged_trace(original)
    assert privileged.messages[0].content == (
        "question\n\n" + HINT_TEMPLATE.format(action="first action")
    )
    assert privileged.messages[2].content == (
        "observation\n\n" + HINT_TEMPLATE.format(action="second action")
    )
    assert privileged.messages[1] == original.messages[1]
    assert privileged.messages[3] == original.messages[3]
    assert all("<HINT>" not in message.content for message in original.messages)


def test_missing_carrier_and_empty_action_fail_explicitly() -> None:
    missing = _unsafe_trace(
        (Message("system", "system"), Message("assistant", "action"))
    )
    with pytest.raises(ValueError, match="carrier"):
        build_privileged_trace(missing)
    empty = _unsafe_trace((Message("user", "question"), Message("assistant", "")))
    with pytest.raises(ValueError, match="non-empty"):
        build_privileged_trace(empty)


def test_only_named_extractor_is_accepted() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        build_privileged_trace(_two_turn_trace(), extractor="implicit_protocol_guess")
