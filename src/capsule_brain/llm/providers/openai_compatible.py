from __future__ import annotations

import os
import time
from typing import Any

import httpx

from capsule_brain.llm.errors import LLMConfigurationError, LLMProviderError
from capsule_brain.llm.models import LLMRequest, LLMResult
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

        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})

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
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError("Malformed provider response") from exc

        usage = data.get("usage", {})
        return LLMResult(
            text=text or "",
            model=model_name,
            provider=self.name,
            latency_ms=latency_ms,
            usage={k: int(v) for k, v in usage.items() if isinstance(v, int)},
            raw=data,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()
