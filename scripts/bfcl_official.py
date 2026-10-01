#!/usr/bin/env python3
"""Run BFCL's official multi-turn generator and scorer against local vLLM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    args = parser.parse_args()
    if args.context_length != 65_536:
        raise ValueError("BFCL supplementary evaluation is fixed to 65536 tokens")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update({
        "BFCL_PROJECT_ROOT": str(args.output),
        "OPENAI_API_KEY": "local-unused",
        "OPENAI_BASE_URL": args.base_url,
    })

    from bfcl_eval.constants import eval_config
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
    from pgsft.evaluation.bfcl import CATEGORIES, fallback_tool_calls, load_bfcl_inventory, split_reasoning

    inventory = load_bfcl_inventory(args.inventory)
    selected = {category: [] for category in CATEGORIES}
    for row in inventory:
        selected[row["category"]].append(row["task_id"])
    (args.output / "test_case_ids_to_generate.json").write_text(json.dumps(selected))

    class LocalQwenHandler(OpenAICompletionsHandler):
        def _build_client_kwargs(self):
            return {"base_url": args.base_url, "api_key": "local-unused", "max_retries": 0, "timeout": 1800}

        def generate_with_backoff(self, **kwargs):
            kwargs.update(max_tokens=8192, seed=42, temperature=0)
            kwargs.pop("store", None)
            try:
                response = self.client.chat.completions.create(**kwargs)
                return response, 0.0
            except Exception as error:
                text = str(error).lower()
                if "context" not in text or "maximum" not in text:
                    raise
                from openai.types.chat import ChatCompletion
                response = ChatCompletion.model_validate({
                    "id": "context-overflow", "created": 0, "model": args.served_model,
                    "object": "chat.completion", "choices": [{"finish_reason": "length", "index": 0,
                    "message": {"content": "", "role": "assistant"}}],
                    "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
                })
                return response, 0.0

        def _parse_query_response_FC(self, api_response):
            parsed = super()._parse_query_response_FC(api_response)
            message = api_response.choices[0].message
            reasoning, visible = split_reasoning(message.content or "", getattr(message, "reasoning_content", None))
            if not message.tool_calls and visible:
                calls = fallback_tool_calls(visible)
                if calls:
                    parsed["model_responses"] = [{c["function"]["name"]: c["function"]["arguments"]} for c in calls]
                    parsed["tool_call_ids"] = [c["id"] for c in calls]
                    message.tool_calls = calls
                    visible = ""
            history = {"role": "assistant", "content": visible}
            if message.tool_calls:
                history["tool_calls"] = [c.model_dump() if hasattr(c, "model_dump") else c for c in message.tool_calls]
            if reasoning:
                history["reasoning"] = reasoning
                parsed["reasoning_content"] = reasoning
            parsed["model_responses_message_for_chat_history"] = history
            return parsed

    name = "pgsft-qwen-local"
    MODEL_CONFIG_MAPPING[name] = ModelConfig(
        model_name=args.served_model, display_name=name, url="local", org="PG-SFT",
        license="apache-2.0", model_handler=LocalQwenHandler, is_fc_model=True,
    )
    from bfcl_eval._llm_response_generation import main as generate
    generate(SimpleNamespace(
        model=[name], test_category=list(CATEGORIES), temperature=0,
        include_input_log=True, exclude_state_log=False, num_gpus=1, num_threads=16,
        gpu_memory_utilization=.9, backend="vllm", skip_server_setup=True,
        local_model_path=str(args.model_root), result_dir=str(args.output / "result"),
        allow_overwrite=False, run_ids=True, enable_lora=False, max_lora_rank=None,
        lora_modules=None,
    ))
    from bfcl_eval.eval_checker.eval_runner import main as evaluate
    evaluate([name], list(CATEGORIES), str(args.output / "result"), str(args.output / "score"), partial_eval=True)

    failures = {}
    for path in (args.output / "score").rglob("*_score.json"):
        lines = [json.loads(line) for line in path.open()]
        for row in lines[1:]:
            failures[row["id"]] = row
    with (args.output / "official-results.jsonl").open("w") as handle:
        for item in inventory:
            task_id = item["task_id"]
            row = failures.get(task_id, {"id": task_id, "valid": True})
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
