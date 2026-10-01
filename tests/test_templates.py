from __future__ import annotations

from pathlib import Path

import pytest
from transformers import AutoTokenizer

from pgsft.model_profiles import MODEL_KEYS, load_model_profile


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "qwen3": {
        "model": "Qwen/Qwen3-4B-Thinking-2507",
        "revision": "768f209d9ea81521153ed38c47d515654e938aea",
        "student": 65_536,
        "privileged": 83_968,
    },
    "qwen35": {
        "model": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "student": 65_536,
        "privileged": 84_992,
    },
}


def _tokenizer(key: str):
    profile = load_model_profile(key, ROOT)
    try:
        return AutoTokenizer.from_pretrained(
            profile.config.name,
            revision=profile.config.revision,
            local_files_only=True,
            trust_remote_code=False,
        )
    except OSError as error:
        pytest.fail(
            "pinned tokenizer snapshot is not cached; fetch the exact model/revision "
            f"before offline verification: {profile.config.name}@{profile.config.revision}: "
            f"{error}"
        )


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_profile_pins_model_revision_lengths_and_template_hashes(key: str) -> None:
    profile = load_model_profile(key, ROOT)
    expected = EXPECTED[key]
    assert profile.schema_version == 1
    assert profile.config.name == expected["model"]
    assert profile.config.revision == expected["revision"]
    assert profile.config.student_max_length == expected["student"]
    assert profile.config.privileged_max_length == expected["privileged"]
    assert len(profile.train_template_record.sha256) == 64
    assert len(profile.inference_template_record.sha256) == 64
    profile.verify_templates(root=ROOT)


@pytest.mark.parametrize(
    ("key", "source_train", "source_inference"),
    [
        (
            "qwen3",
            "qwen3_thinking_train.jinja",
            "qwen3_thinking_inference.jinja",
        ),
        (
            "qwen35",
            "qwen3_5_full_think_train.jinja",
            "qwen3_5_full_think_inference.jinja",
        ),
    ],
)
def test_release_templates_are_byte_exact_approved_copies(
    key: str, source_train: str, source_inference: str
) -> None:
    profile = load_model_profile(key, ROOT)
    source_root = ROOT.parents[1] / "configs" / "chat_templates"
    assert profile.config.train_template.read_bytes() == (
        source_root / source_train
    ).read_bytes()
    assert profile.config.inference_template.read_bytes() == (
        source_root / source_inference
    ).read_bytes()


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_train_template_masks_only_complete_assistant_segments(key: str) -> None:
    profile = load_model_profile(key, ROOT)
    tokenizer = _tokenizer(key)
    messages = [
        {"role": "system", "content": "SYSTEM_SENTINEL"},
        {"role": "user", "content": "USER_SENTINEL"},
        {
            "role": "assistant",
            "reasoning_content": "REASONING_SENTINEL",
            "content": "ACTION_SENTINEL",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "inspect",
                        "arguments": {"path": "result.json"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"TOOL_SENTINEL":"<think> deceptive </tool_call>"}',
        },
        {
            "role": "assistant",
            "reasoning_content": "FINAL_REASONING",
            "content": "FINAL_ACTION",
        },
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        chat_template=profile.config.train_template.read_text(encoding="utf-8"),
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    masked = tokenizer.decode(
        [
            token
            for token, selected in zip(
                encoded["input_ids"], encoded["assistant_masks"], strict=True
            )
            if selected
        ],
        skip_special_tokens=False,
    )
    for expected in (
        "<think>",
        "REASONING_SENTINEL",
        "ACTION_SENTINEL",
        "<tool_call>",
        "</tool_call>",
        "FINAL_REASONING",
        "FINAL_ACTION",
        "<|im_end|>",
    ):
        assert expected in masked
    for excluded in ("SYSTEM_SENTINEL", "USER_SENTINEL", "TOOL_SENTINEL"):
        assert excluded not in masked


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_inference_prefix_and_hint_non_invention(key: str) -> None:
    profile = load_model_profile(key, ROOT)
    tokenizer = _tokenizer(key)
    template = profile.config.inference_template.read_text(encoding="utf-8")
    messages = [{"role": "user", "content": "Solve this."}]
    rendered = tokenizer.apply_chat_template(
        messages,
        chat_template=template,
        tokenize=False,
        add_generation_prompt=True,
    )
    assert rendered.endswith("<|im_start|>assistant\n<think>\n")
    assert "<HINT>" not in rendered

    hinted = tokenizer.apply_chat_template(
        [{"role": "user", "content": "<HINT>explicit</HINT>"}],
        chat_template=template,
        tokenize=False,
        add_generation_prompt=True,
    )
    assert "<HINT>explicit</HINT>" in hinted
