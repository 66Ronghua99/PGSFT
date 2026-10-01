from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest
from transformers import AutoTokenizer

from pgsft.data.canonical import CanonicalTrace, Message
from pgsft.data.tokenization import (
    IGNORE_INDEX,
    TokenizationError,
    tokenize_pgsft_trace,
    tokenize_sft_trace,
)
from pgsft.model_profiles import MODEL_KEYS, load_model_profile


ROOT = Path(__file__).resolve().parents[2]


def _trace() -> CanonicalTrace:
    return CanonicalTrace(
        id="two-turn",
        messages=(
            Message("system", "system"),
            Message("user", "question"),
            Message(
                "assistant",
                '{"literal":"<think> and </tool_call>","action":"inspect"}',
                "reason one",
            ),
            Message("tool", '{"result":"literal <think> and </tool_call>"}'),
            Message("assistant", "final action", "reason two"),
        ),
    )


class FakeTokenizer:
    def __init__(self, *, mismatch_with_hint: bool = False, prefix: bool = True):
        self.mismatch_with_hint = mismatch_with_hint
        self.prefix = prefix

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        **_: Any,
    ):
        rendered = json.dumps(messages, sort_keys=True, separators=(",", ":"))
        if not tokenize:
            return rendered
        hinted = "<HINT>" in rendered
        ids = [7] if self.prefix else []
        mask = [0] * len(ids)
        assistant_index = 0
        for message in messages:
            if message["role"] == "assistant":
                tokens = [100 + assistant_index, 200 + assistant_index]
                if hinted and self.mismatch_with_hint:
                    tokens[-1] += 1
                ids.extend(tokens)
                mask.extend([1, 1])
                assistant_index += 1
            else:
                width = 1 + len(message["content"]) // 10
                ids.extend([20 + width] * width)
                mask.extend([0] * width)
        return {"input_ids": ids, "assistant_masks": mask}


def _profile_with_limits(student: int, privileged: int):
    profile = load_model_profile("qwen35", ROOT)
    return replace(
        profile,
        config=replace(
            profile.config,
            student_max_length=student,
            privileged_max_length=privileged,
        ),
    )


def _unsafe_assistant_first() -> CanonicalTrace:
    trace = object.__new__(CanonicalTrace)
    object.__setattr__(trace, "id", "position-zero")
    object.__setattr__(trace, "messages", (Message("assistant", "action"),))
    object.__setattr__(trace, "tools", ())
    object.__setattr__(trace, "source", None)
    return trace


def test_sft_labels_every_complete_assistant_run_with_dense_turn_ids() -> None:
    row = tokenize_sft_trace(
        _trace(),
        tokenizer=FakeTokenizer(),
        profile=_profile_with_limits(1_000, 2_000),
    )
    active = [index for index, label in enumerate(row.labels) if label != IGNORE_INDEX]
    assert len(active) == 4
    assert [row.assistant_turn_ids[index] for index in active] == [0, 0, 1, 1]
    assert all(
        row.assistant_turn_ids[index] == IGNORE_INDEX
        for index, label in enumerate(row.labels)
        if label == IGNORE_INDEX
    )
    assert row.length == len(row.input_ids)
    assert len(row.rendered_sha256) == 64


def test_paired_targets_use_semantic_keys_at_different_offsets() -> None:
    row = tokenize_pgsft_trace(
        _trace(),
        tokenizer=FakeTokenizer(),
        profile=_profile_with_limits(1_000, 2_000),
    )
    assert row.turn_indices == (0, 0, 1, 1)
    assert row.token_indices == (0, 1, 0, 1)
    assert row.target_ids == (100, 200, 101, 201)
    assert row.student_positions != row.privileged_positions
    for position, target in zip(row.student_positions, row.target_ids, strict=True):
        assert row.student_input_ids[position + 1] == target
    for position, target in zip(row.privileged_positions, row.target_ids, strict=True):
        assert row.privileged_input_ids[position + 1] == target


def test_privileged_only_overflow_is_rejected_without_partial_alignment() -> None:
    tokenizer = FakeTokenizer()
    baseline = tokenize_sft_trace(
        _trace(), tokenizer=tokenizer, profile=_profile_with_limits(1_000, 2_000)
    )
    profile = _profile_with_limits(baseline.length, baseline.length)
    with pytest.raises(TokenizationError, match="exceeds limit"):
        tokenize_pgsft_trace(_trace(), tokenizer=tokenizer, profile=profile)


def test_target_mismatch_and_position_zero_are_rejected() -> None:
    with pytest.raises(TokenizationError, match="target token mismatch"):
        tokenize_pgsft_trace(
            _trace(),
            tokenizer=FakeTokenizer(mismatch_with_hint=True),
            profile=_profile_with_limits(1_000, 2_000),
        )
    with pytest.raises(TokenizationError, match="position zero"):
        tokenize_sft_trace(
            _unsafe_assistant_first(),
            tokenizer=FakeTokenizer(prefix=False),
            profile=_profile_with_limits(1_000, 2_000),
        )


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_pinned_tokenizers_align_full_assistant_targets(key: str) -> None:
    profile = load_model_profile(key, ROOT)
    tokenizer = AutoTokenizer.from_pretrained(
        profile.config.name,
        revision=profile.config.revision,
        local_files_only=True,
        trust_remote_code=False,
    )
    sft = tokenize_sft_trace(_trace(), tokenizer=tokenizer, profile=profile)
    paired = tokenize_pgsft_trace(_trace(), tokenizer=tokenizer, profile=profile)
    assert paired.target_ids == tuple(
        label for label in sft.labels if label != IGNORE_INDEX
    )
    assert set(paired.turn_indices) == {0, 1}
    assert paired.privileged_length > paired.student_length
    assert paired.student_rendered_sha256 == sft.rendered_sha256
