from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from capsule_brain.llm.models import LLMRequest, LLMResult


class LLMProvider(ABC):
    name: str

    @abstractmethod
    async def generate(self, request: LLMRequest, model_cfg: dict) -> LLMResult:
        raise NotImplementedError

    async def stream(
        self,
        request: LLMRequest,
        model_cfg: dict,
    ) -> AsyncIterator[str]:
        # Providers can override with native streaming.
        result = await self.generate(request, model_cfg)
        yield result.text

    async def close(self) -> None:
        return None
