from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from .models import MemoryRecord


class MemoryRepository(ABC):
    @abstractmethod
    async def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def create(self, record: MemoryRecord) -> MemoryRecord:
        raise NotImplementedError

    @abstractmethod
    async def get(self, memory_id: str) -> MemoryRecord | None:
        raise NotImplementedError

    @abstractmethod
    async def recent(
        self,
        *,
        limit: int = 100,
        include_archived: bool = False,
    ) -> list[MemoryRecord]:
        raise NotImplementedError

    @abstractmethod
    async def update(self, record: MemoryRecord) -> MemoryRecord:
        raise NotImplementedError

    @abstractmethod
    async def archive(self, memory_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def count(self, *, include_archived: bool = False) -> int:
        raise NotImplementedError
