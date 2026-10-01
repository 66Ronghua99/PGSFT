"""Text tool-call framing. JSON string contents never act as protocol delimiters."""

import json
import re


_DECODER = json.JSONDecoder()
OPEN = "<tool_call>"
CLOSE = "</tool_call>"


def _tool_action_segment(content: str, *, thinking_open: bool = False) -> str:
    """Locate the first thinking closure outside complete JSON objects.

    The caller supplies the template's initial state: the generation prompt's
    implicit <think> is absent from the returned completion text.
    """
    offset = 0
    while offset < len(content):
        if content.startswith("</think>", offset):
            return content[offset + len("</think>"):]
        if content.startswith("<think>", offset):
            thinking_open = True
            offset += len("<think>")
            continue
        if not thinking_open and content.startswith(OPEN, offset):
            return content
        if content[offset] == "{":
            try:
                _, end = _DECODER.raw_decode(content, offset)
            except json.JSONDecodeError:
                pass
            else:
                if not thinking_open:
                    return content
                offset = end
                continue
        offset += 1
    return "" if thinking_open else content


def extract_tool_calls(
    content: str, *, thinking_open: bool = False,
) -> tuple[list[dict], int, dict | None, str | None]:
    """Return valid calls, block count, final call and final framing error.

    Only a final complete block may execute; an incomplete final opening also
    invalidates earlier calls. Recovery is limited to an anchored JSON object
    with its closing marker and no opening marker.
    """
    action = _tool_action_segment(content, thinking_open=thinking_open).strip()
    if not action:
        return [], 0, None, None
    if action.startswith("{"):
        try:
            candidate, end = _DECODER.raw_decode(action)
        except json.JSONDecodeError:
            pass
        else:
            if action[end:].strip() == CLOSE:
                action = OPEN + action

    calls = []
    count = 0
    last = None
    error = None
    offset = 0
    while True:
        start = action.find(OPEN, offset)
        if start < 0:
            if "<think>" in action[offset:] or "</think>" in action[offset:]:
                return calls, count + 1, None, "Unexpected thinking delimiter in action"
            break
        # A second thinking region is not executable action text.
        if "<think>" in action[offset:start] or "</think>" in action[offset:start]:
            return calls, count + 1, None, "Unexpected thinking delimiter in action"
        count += 1
        last = None
        body_start = start + len(OPEN)
        while body_start < len(action) and action[body_start].isspace():
            body_start += 1
        fence = re.match(r"```(?:json)?\s*", action[body_start:])
        if fence:
            body_start += fence.end()
        try:
            data, end = _DECODER.raw_decode(action, body_start)
        except json.JSONDecodeError as invalid:
            error = str(invalid)
            closing = action.find(CLOSE, body_start)
            if closing < 0:
                break
            offset = closing + len(CLOSE)
            continue
        suffix = re.match(r"\s*" + (r"```\s*" if fence else "") + re.escape(CLOSE), action[end:])
        if suffix is None:
            error = "Expected </tool_call> after the complete JSON object"
            # JSON was fully consumed, so searching from its end cannot mistake
            # a literal opening tag in an argument for a new action.
            offset = end
            continue
        offset = end + suffix.end()
        if not isinstance(data, dict) or not isinstance(data.get("name"), str):
            error = "Tool call must be an object with a string name"
            continue
        calls.append(data)
        last = data
        error = None
    return calls, count, last, error
