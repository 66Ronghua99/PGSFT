"""Validation and calls for an existing OpenAI-compatible endpoint."""

from __future__ import annotations

import json
from pathlib import Path
import urllib.error
import urllib.request
from typing import Any


class EndpointError(RuntimeError):
    pass


class ContextOverflowError(EndpointError):
    pass


class InfrastructureEndpointError(EndpointError):
    pass


class OpenAIEndpoint:
    def __init__(self, base_url: str, served_name: str, model_root: Path):
        if not base_url.rstrip("/").endswith("/v1"):
            raise ValueError("OpenAI endpoint URL must end in /v1")
        self.base_url = base_url.rstrip("/")
        self.served_name = served_name
        self.model_root = model_root.resolve(strict=True)

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if data is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                value = json.loads(response.read())
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            if error.code == 400 and "context" in body.lower():
                raise ContextOverflowError(body) from error
            raise EndpointError(f"endpoint HTTP {error.code}: {body}") from error
        except OSError as error:
            raise InfrastructureEndpointError(f"endpoint transport failed: {error}") from error
        except json.JSONDecodeError as error:
            raise EndpointError(f"endpoint request failed: {error}") from error
        if not isinstance(value, dict):
            raise EndpointError("endpoint returned a non-object response")
        return value

    def validate(self) -> dict[str, Any]:
        value = self._request("/models")
        rows = value.get("data")
        if not isinstance(rows, list) or len(rows) != 1 or rows[0].get("id") != self.served_name:
            raise EndpointError("/v1/models must expose exactly the configured served name")
        root_value = rows[0].get("root", rows[0].get("model_root"))
        if not isinstance(root_value, str) or Path(root_value).resolve() != self.model_root:
            raise EndpointError("served model root does not match requested merged model")
        return rows[0]

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.served_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        value = self._request("/chat/completions", payload)
        choices = value.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise EndpointError("completion must contain exactly one choice")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise EndpointError("completion is missing string content")
        if choice.get("finish_reason") not in {"stop", "length"}:
            raise EndpointError("completion has an unsupported finish reason")
        return {"content": message["content"], "finish_reason": choice["finish_reason"], "usage": value.get("usage")}
