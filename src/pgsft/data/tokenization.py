"""Complete-assistant SFT and paired PG-SFT tokenization contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Sequence

from pgsft.data.canonical import CanonicalTrace
from pgsft.data.privileged import build_privileged_trace
from pgsft.model_profiles import ModelProfile


IGNORE_INDEX = -100


class TokenizationError(ValueError):
    """A trace cannot satisfy the immutable tokenization contract."""


@dataclass(frozen=True)
class SFTTokenizedRow:
    id: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    assistant_turn_ids: tuple[int, ...]
    length: int
    rendered_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PGSFTTokenizedRow:
    id: str
    student_input_ids: tuple[int, ...]
    privileged_input_ids: tuple[int, ...]
    student_positions: tuple[int, ...]
    privileged_positions: tuple[int, ...]
    target_ids: tuple[int, ...]
    turn_indices: tuple[int, ...]
    token_indices: tuple[int, ...]
    student_length: int
    privileged_length: int
    student_rendered_sha256: str
    privileged_rendered_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Lane:
    id: str
    input_ids: tuple[int, ...]
    assistant_mask: tuple[int, ...]
    keys: tuple[tuple[int, int], ...]
    target_positions: tuple[int, ...]
    target_ids: tuple[int, ...]
    rendered_sha256: str


def _template_messages(trace: CanonicalTrace) -> list[dict[str, Any]]:
    messages = []
    for message in trace.messages:
        value = message.to_dict()
        if message.role == "assistant" and message.reasoning_content is None:
            # Literal protocol markers in JSON action strings are action content.
            value["reasoning_content"] = ""
        messages.append(value)
    return messages


def _template_kwargs(trace: CanonicalTrace, template: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "chat_template": template,
        "add_generation_prompt": False,
    }
    if trace.tools:
        kwargs["tools"] = [dict(tool) for tool in trace.tools]
    return kwargs


def _runs(mask: Sequence[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate([*mask, 0]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append((start, index))
            start = None
    return runs


def _tokenize_lane(
    trace: CanonicalTrace,
    *,
    tokenizer: Any,
    template: str,
    max_length: int,
) -> _Lane:
    if max_length <= 0:
        raise TokenizationError("max_length must be positive")
    messages = _template_messages(trace)
    kwargs = _template_kwargs(trace, template)
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, **kwargs
    )
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        **kwargs,
    )
    input_ids = tuple(int(token) for token in encoded["input_ids"])
    assistant_mask = tuple(int(active) for active in encoded["assistant_masks"])
    if len(input_ids) != len(assistant_mask):
        raise TokenizationError(f"{trace.id}: input/mask length mismatch")
    if len(input_ids) > max_length:
        raise TokenizationError(
            f"{trace.id}: lane length {len(input_ids)} exceeds limit {max_length}"
        )
    assistant_count = sum(message.role == "assistant" for message in trace.messages)
    runs = _runs(assistant_mask)
    if len(runs) != assistant_count:
        raise TokenizationError(
            f"{trace.id}: assistant runs {len(runs)} != turns {assistant_count}"
        )
    keys: list[tuple[int, int]] = []
    positions: list[int] = []
    targets: list[int] = []
    for turn_index, (start, end) in enumerate(runs):
        if start >= end:
            raise TokenizationError(f"{trace.id}: assistant turn {turn_index} is empty")
        for token_index, target_position in enumerate(range(start, end)):
            prediction_position = target_position - 1
            if prediction_position < 0:
                raise TokenizationError(
                    f"{trace.id}: assistant target at position zero has no predictor"
                )
            keys.append((turn_index, token_index))
            positions.append(prediction_position)
            targets.append(input_ids[target_position])
    if set(turn for turn, _ in keys) != set(range(assistant_count)):
        raise TokenizationError(f"{trace.id}: assistant turn indices are not dense")
    if any(left >= right for left, right in zip(positions, positions[1:])):
        raise TokenizationError(f"{trace.id}: prediction positions are not increasing")
    for position, target in zip(positions, targets, strict=True):
        if position + 1 >= len(input_ids) or input_ids[position + 1] != target:
            raise TokenizationError(f"{trace.id}: invalid causal target at {position}")
    return _Lane(
        id=trace.id,
        input_ids=input_ids,
        assistant_mask=assistant_mask,
        keys=tuple(keys),
        target_positions=tuple(positions),
        target_ids=tuple(targets),
        rendered_sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    )


def tokenize_sft_trace(
    trace: CanonicalTrace,
    *,
    tokenizer: Any,
    profile: ModelProfile,
    max_length: int | None = None,
) -> SFTTokenizedRow:
    lane = _tokenize_lane(
        trace,
        tokenizer=tokenizer,
        template=profile.config.train_template.read_text(encoding="utf-8"),
        max_length=max_length or profile.config.student_max_length,
    )
    labels = tuple(
        token if active else IGNORE_INDEX
        for token, active in zip(lane.input_ids, lane.assistant_mask, strict=True)
    )
    turn_ids = [IGNORE_INDEX] * len(lane.input_ids)
    for (turn_index, _), prediction_position in zip(
        lane.keys, lane.target_positions, strict=True
    ):
        turn_ids[prediction_position + 1] = turn_index
    return SFTTokenizedRow(
        id=trace.id,
        input_ids=lane.input_ids,
        labels=labels,
        assistant_turn_ids=tuple(turn_ids),
        length=len(lane.input_ids),
        rendered_sha256=lane.rendered_sha256,
    )


def tokenize_pgsft_trace(
    trace: CanonicalTrace,
    *,
    tokenizer: Any,
    profile: ModelProfile,
) -> PGSFTTokenizedRow:
    privileged_limit = profile.config.privileged_max_length
    if privileged_limit is None:
        raise TokenizationError(f"{profile.key}: privileged_max_length is required")
    template = profile.config.train_template.read_text(encoding="utf-8")
    privileged = build_privileged_trace(trace)
    student_lane = _tokenize_lane(
        trace,
        tokenizer=tokenizer,
        template=template,
        max_length=profile.config.student_max_length,
    )
    privileged_lane = _tokenize_lane(
        privileged,
        tokenizer=tokenizer,
        template=template,
        max_length=privileged_limit,
    )
    if student_lane.keys != privileged_lane.keys:
        raise TokenizationError(f"{trace.id}: paired target identity mismatch")
    if student_lane.target_ids != privileged_lane.target_ids:
        raise TokenizationError(f"{trace.id}: paired target token mismatch")
    turn_indices = tuple(key[0] for key in student_lane.keys)
    token_indices = tuple(key[1] for key in student_lane.keys)
    count = len(student_lane.target_ids)
    arrays = (
        student_lane.target_positions,
        privileged_lane.target_positions,
        turn_indices,
        token_indices,
    )
    if count == 0 or any(len(values) != count for values in arrays):
        raise TokenizationError(f"{trace.id}: invalid paired target arrays")
    return PGSFTTokenizedRow(
        id=trace.id,
        student_input_ids=student_lane.input_ids,
        privileged_input_ids=privileged_lane.input_ids,
        student_positions=student_lane.target_positions,
        privileged_positions=privileged_lane.target_positions,
        target_ids=student_lane.target_ids,
        turn_indices=turn_indices,
        token_indices=token_indices,
        student_length=len(student_lane.input_ids),
        privileged_length=len(privileged_lane.input_ids),
        student_rendered_sha256=student_lane.rendered_sha256,
        privileged_rendered_sha256=privileged_lane.rendered_sha256,
    )
