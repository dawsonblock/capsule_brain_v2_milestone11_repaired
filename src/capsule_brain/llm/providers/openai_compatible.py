from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from capsule_brain.llm.errors import LLMConfigurationError, LLMProviderError
from capsule_brain.llm.models import LLMRequest, LLMResult
from capsule_brain.llm.tools import ToolCall
from .base import LLMProvider


class OpenAICompatibleProvider(LLMProvider):
    name = "openai"

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._owns_client = client is None
        # Disable httpx's default 5s read timeout. Request timeout is managed
        # at the gateway level via asyncio.wait_for; without this override,
        # longer generations silently abort with httpx.ReadTimeout.
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(None))

    async def generate(self, request: LLMRequest, model_cfg: dict[str, Any]) -> LLMResult:
        api_base = str(model_cfg.get("api_base", "https://api.openai.com/v1")).rstrip("/")
        model_name = str(model_cfg.get("model_name", "")).strip()
        key_env = str(model_cfg.get("api_key_env", "OPENAI_API_KEY"))
        api_key = os.getenv(key_env)

        if not model_name:
            raise LLMConfigurationError("model_name is required")
        if not api_key:
            raise LLMConfigurationError(f"Missing API key environment variable: {key_env}")

        messages = self._build_messages(request)

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": request.temperature if request.temperature is not None
                else model_cfg.get("temperature", 0.3),
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.response_format == "json":
            payload["response_format"] = {"type": "json_object"}
        if request.tools:
            payload["tools"] = [spec.to_openai_schema() for spec in request.tools]
            # Allow the model to choose when to call tools rather than forcing
            # or forbidding tool use. Callers can override by setting
            # ``tool_choice`` in request.metadata if needed.
            payload.setdefault("tool_choice", "auto")

        started = time.perf_counter()
        response = await self.client.post(
            f"{api_base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        latency_ms = (time.perf_counter() - started) * 1000

        if response.is_error:
            raise LLMProviderError(
                f"{self.name} returned HTTP {response.status_code}: {response.text[:500]}"
            )

        data = response.json()
        try:
            choice = data["choices"][0]
            message = choice["message"]
            text = message.get("content") or ""
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError("Malformed provider response") from exc

        tool_calls = self._parse_tool_calls(message)

        usage = data.get("usage", {})
        return LLMResult(
            text=text or "",
            model=model_name,
            provider=self.name,
            latency_ms=latency_ms,
            usage={k: int(v) for k, v in usage.items() if isinstance(v, int)},
            raw=data,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _build_messages(request: LLMRequest) -> list[dict[str, Any]]:
        """Build the OpenAI chat messages list from an LLMRequest.

        Includes the system prompt, the user prompt, and any prior tool
        results (so multi-turn tool-calling loops can continue). Tool results
        are appended as ``tool`` role messages per the OpenAI spec.
        """
        messages: list[dict[str, Any]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        for result in request.tool_results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.get("tool_call_id", ""),
                    "content": str(result.get("content", "")),
                }
            )
        return messages

    @staticmethod
    def _parse_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
        """Parse ``message.tool_calls`` into typed ToolCall objects.

        Defends against malformed arguments: if the model returns invalid
        JSON for a tool's arguments, we substitute an empty dict and let the
        ToolRegistry's required-key validation surface a clear error rather
        than crashing the gateway.
        """
        raw_calls = message.get("tool_calls") or []
        calls: list[ToolCall] = []
        for raw in raw_calls:
            try:
                call_id = str(raw.get("id", ""))
                function = raw.get("function") or {}
                name = str(function.get("name", ""))
                args_blob = function.get("arguments", "{}")
                try:
                    arguments = json.loads(args_blob) if args_blob else {}
                except json.JSONDecodeError:
                    arguments = {}
                if name:
                    calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
            except Exception:
                continue
        return calls

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()
