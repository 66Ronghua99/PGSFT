"""Pinned local model loading and LoRA attachment."""

from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model

from pgsft.model_profiles import ModelProfile


def load_lora_model(
    profile: ModelProfile,
    *,
    attention_implementation: str,
    gradient_checkpointing: bool,
) -> torch.nn.Module:
    """Load the exact local snapshot; never probe or change attention backends."""
    if attention_implementation not in {"sdpa", "flash_attention_2", "flash_attention_4"}:
        raise ValueError(f"unsupported attention implementation: {attention_implementation}")
    snapshot = Path(
        snapshot_download(
            repo_id=profile.config.name,
            revision=profile.config.revision,
            local_files_only=True,
        )
    ).resolve()
    if profile.key == "qwen3":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation=attention_implementation,
        )
    elif profile.key == "qwen35":
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            snapshot,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation=attention_implementation,
        )
    else:
        raise ValueError(f"unsupported model profile: {profile.key}")
    observed = getattr(model.config, "_attn_implementation", None)
    if observed != attention_implementation:
        raise RuntimeError(
            f"attention backend mismatch: requested={attention_implementation} observed={observed}"
        )
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    lora = get_peft_model(
        model,
        LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.0,
            target_modules="all-linear",
            exclude_modules=["lm_head"],
            task_type="CAUSAL_LM",
        ),
    )
    trainable = [name for name, parameter in lora.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("trainable parameter inventory is not LoRA-only")
    if any("lm_head" in name for name in trainable):
        raise RuntimeError("output head must remain frozen")
    return lora
