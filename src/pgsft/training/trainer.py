"""Objective runner and formal single-node/DDP training loop."""

from __future__ import annotations

from contextlib import nullcontext
import json
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_scheduler

from pgsft.methods.base import MethodObjective, ObjectiveContext, ObjectiveOutput
from pgsft.methods.registry import get_method
from pgsft.methods.psft import PSFTObjective
from pgsft.model_profiles import load_model_profile
from pgsft.training.checkpoint import select_checkpoint
from pgsft.training.collator import PairedCollator, SFTCollator
from pgsft.training.model import load_lora_model
from pgsft.training.normalization import GlobalTokenNormalizer
from pgsft.training.snapshot import trainable_adapter_state


class PGSFTTrainer:
    def __init__(self, objective: MethodObjective):
        self.objective = objective
        self.normalizer = GlobalTokenNormalizer()

    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        output_weight: torch.Tensor,
        output_bias: torch.Tensor | None = None,
        extras: dict | None = None,
    ) -> ObjectiveOutput:
        divisor = batch.get("global_token_divisor")
        context = ObjectiveContext(
            output_weight=output_weight,
            output_bias=output_bias,
            global_token_divisor=divisor,
            extras={} if extras is None else extras,
        )
        return self.objective.compute(batch, context)


class _Rows(Dataset):
    def __init__(self, path: Path):
        with path.open(encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def _decoder_hidden(model: torch.nn.Module, profile_key: str, ids, mask):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if profile_key == "qwen3":
        decoder = base.model
    else:
        decoder = base.model.language_model
    output = decoder(
        input_ids=ids,
        attention_mask=mask,
        use_cache=False,
        return_dict=True,
    )
    return output.last_hidden_state


def _output_weight(model: torch.nn.Module) -> torch.Tensor:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    weight = base.lm_head.weight
    if weight.requires_grad:
        raise RuntimeError("formal objectives require a frozen output head")
    return weight


def _active_hidden(hidden: torch.Tensor, labels: torch.Tensor):
    active = labels[:, 1:] != -100
    return hidden[:, :-1][active], labels[:, 1:][active]


def _paired_hidden(hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
    return hidden[batch, positions].reshape(-1, hidden.shape[-1])


def _teacher_hidden(model, profile_key, ids, mask):
    previous = model.training
    model.eval()
    try:
        with model.disable_adapter(), torch.no_grad():
            return _decoder_hidden(model, profile_key, ids, mask).detach()
    finally:
        model.train(previous)


def _target_logp(hidden, weight, targets):
    values = []
    for start in range(0, targets.numel(), 128):
        end = min(start + 128, targets.numel())
        logits = hidden[start:end].float() @ weight.detach().float().t()
        values.append(
            torch.log_softmax(logits, dim=-1)
            .gather(1, targets[start:end, None])
            .squeeze(1)
        )
    return torch.cat(values)


def _old_policy_logp(model, profile_key, batch, weight, snapshot):
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name in snapshot.tensors
    }
    if set(parameters) != set(snapshot.tensors):
        raise RuntimeError("old-policy snapshot parameter inventory drift")
    current = {name: parameter.detach().clone() for name, parameter in parameters.items()}
    try:
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(snapshot.tensors[name].to(parameter.device, parameter.dtype))
            hidden = _decoder_hidden(
                model, profile_key, batch["input_ids"], batch["attention_mask"]
            )
            active, targets = _active_hidden(hidden, batch["labels"])
            return _target_logp(active, weight, targets).detach()
    finally:
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(current[name])


def _objective_batch(method, model, profile_key, batch, weight):
    if method.required_row_schema == "sft":
        hidden = _decoder_hidden(
            model, profile_key, batch["input_ids"], batch["attention_mask"]
        )
        if method.name in {"sft", "dft"}:
            return {"hidden_states": hidden, "labels": batch["labels"]}
        active, targets = _active_hidden(hidden, batch["labels"])
        if method.name == "sft_base_kl":
            base_hidden = _teacher_hidden(
                model, profile_key, batch["input_ids"], batch["attention_mask"]
            )
            base_active, base_targets = _active_hidden(base_hidden, batch["labels"])
            if not torch.equal(targets, base_targets):
                raise RuntimeError("Base/student target drift")
            return {"student_hidden": active, "base_hidden": base_active, "targets": targets}
        if method.name == "psft":
            if method.snapshot is None:
                raise RuntimeError("PSFT snapshot is not initialized")
            return {
                "student_logp": _target_logp(active, weight, targets),
                "old_logp": _old_policy_logp(
                    model, profile_key, batch, weight, method.snapshot
                ),
            }
    student_hidden = _decoder_hidden(
        model,
        profile_key,
        batch["student_input_ids"],
        batch["student_attention_mask"],
    )
    base_hidden = _teacher_hidden(
        model,
        profile_key,
        batch["student_input_ids"],
        batch["student_attention_mask"],
    )
    privileged_hidden = _teacher_hidden(
        model,
        profile_key,
        batch["privileged_input_ids"],
        batch["privileged_attention_mask"],
    )
    targets = batch["target_ids"].reshape(-1)
    return {
        "student_hidden": _paired_hidden(student_hidden, batch["student_positions"]),
        "base_hidden": _paired_hidden(base_hidden, batch["student_positions"]),
        "privileged_hidden": _paired_hidden(
            privileged_hidden, batch["privileged_positions"]
        ),
        "targets": targets,
        "turn_ids": batch["turn_indices"].reshape(-1),
    }


def _local_count(batch, schema: str) -> int:
    if schema == "sft":
        return int((batch["labels"][:, 1:] != -100).sum())
    return int(batch["target_ids"].numel())


def run_training(resolved: dict, *, run_dir: Path, resume: bool) -> None:
    """Run a formal recipe in the current torchrun/Accelerate process group."""
    from accelerate import Accelerator

    recipe = resolved["recipe"]
    root = Path(__file__).resolve().parents[3]
    profile = load_model_profile(recipe["model_profile"], root)
    method = get_method(recipe["method"])
    training = recipe["training"]
    accelerator = Accelerator(mixed_precision="bf16")
    model = load_lora_model(
        profile,
        attention_implementation=training["attention_implementation"],
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
    )
    resume_state = None
    resume_step = 0
    if resume:
        checkpoints = sorted(
            (
                (int(path.name.removeprefix("checkpoint-")), path)
                for path in run_dir.glob("checkpoint-*")
                if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
            ),
            key=lambda item: item[0],
        )
        if not checkpoints:
            raise ValueError("resume run contains no checkpoint-N directory")
        resume_step, checkpoint = checkpoints[-1]
        state_path = checkpoint / "training_state.pt"
        if not state_path.is_file():
            raise ValueError("resume checkpoint is missing training_state.pt")
        resume_state = torch.load(state_path, map_location="cpu", weights_only=False)
        if resume_state.get("step") != resume_step:
            raise ValueError("resume checkpoint step metadata drift")
        from peft.utils import set_peft_model_state_dict
        from safetensors.torch import load_file

        adapter_path = checkpoint / "adapter_model.safetensors"
        if not adapter_path.is_file():
            raise ValueError("resume checkpoint is missing adapter_model.safetensors")
        result = set_peft_model_state_dict(model, load_file(adapter_path))
        if getattr(result, "unexpected_keys", ()) or getattr(result, "mismatched_keys", ()):
            raise ValueError("resume adapter state is incompatible with the model")
        if isinstance(method, PSFTObjective):
            method.load_state_dict(resume_state["objective"], completed_step=resume_step)
    elif isinstance(method, PSFTObjective):
        method.initialize_snapshot(model)
    tokenizer = AutoTokenizer.from_pretrained(
        profile.config.name,
        revision=profile.config.revision,
        local_files_only=True,
        trust_remote_code=False,
    )
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise RuntimeError("tokenizer has no pad token")
    train_path = root / recipe["data"]["train_jsonl"]
    collator = (
        SFTCollator(pad_id)
        if method.required_row_schema == "sft"
        else PairedCollator(pad_id)
    )
    generator = torch.Generator().manual_seed(int(training["seed"]))
    loader = DataLoader(
        _Rows(train_path),
        batch_size=1,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training["learning_rate"]),
        weight_decay=0.0,
        fused=torch.cuda.is_available(),
    )
    max_steps = int(training["max_steps"])
    scheduler = get_scheduler(
        training["scheduler"],
        optimizer,
        num_warmup_steps=int(training["warmup_steps"]),
        num_training_steps=max_steps,
    )
    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
    unwrapped = accelerator.unwrap_model(model)
    world = accelerator.num_processes
    accumulation = int(training["global_batch_size"]) // world
    if accumulation <= 0 or accumulation * world != int(training["global_batch_size"]):
        raise ValueError("global batch size must be divisible by world size")
    step = resume_step
    model.train()
    iterator = iter(loader)
    # Replay only the sampler traversal, not model work, to restore the exact
    # shuffled position under the fixed generator seed.
    for _ in range(resume_step * accumulation):
        try:
            next(iterator)
        except StopIteration:
            iterator = iter(loader)
            next(iterator)
    while step < max_steps:
        window = []
        for _ in range(accumulation):
            try:
                window.append(next(iterator))
            except StopIteration:
                iterator = iter(loader)
                window.append(next(iterator))
        local_count = sum(_local_count(batch, method.required_row_schema) for batch in window)
        global_count = int(
            accelerator.reduce(
                torch.tensor(local_count, device=accelerator.device), reduction="sum"
            ).item()
        )
        divisor = global_count / world
        optimizer.zero_grad(set_to_none=True)
        metric_sums: dict[str, float] = {}
        for index, batch in enumerate(window):
            synchronization = index == len(window) - 1
            context_manager = nullcontext() if synchronization else accelerator.no_sync(model)
            with context_manager:
                weight = _output_weight(unwrapped)
                objective_batch = _objective_batch(
                    method, unwrapped, profile.key, batch, weight
                )
                output = method.compute(
                    objective_batch,
                    ObjectiveContext(
                        output_weight=weight,
                        global_token_divisor=divisor,
                        extras={"projection_backend": "liger"},
                    ),
                )
                accelerator.backward(output.loss)
            for name, value in output.metrics.items():
                if value.numel() == 1:
                    metric_sums[name] = metric_sums.get(name, 0.0) + float(value)
        accelerator.clip_grad_norm_(model.parameters(), float(training["max_grad_norm"]))
        optimizer.step()
        scheduler.step()
        step += 1
        method.on_optimizer_step_end(step, unwrapped)
        if accelerator.is_main_process:
            with (run_dir / "train.log").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, "metrics": metric_sums}, sort_keys=True) + "\n")
        if step % int(training["save_steps"]) == 0 or step == max_steps:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                checkpoint = run_dir / f"checkpoint-{step}"
                unwrapped.save_pretrained(checkpoint, safe_serialization=True)
                torch.save(
                    {
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": step,
                        "objective": method.state_dict(),
                    },
                    checkpoint / "training_state.pt",
                )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        select_checkpoint(run_dir, recipe["selection"])
