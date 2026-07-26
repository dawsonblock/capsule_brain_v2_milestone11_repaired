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
    async def archive_batch(self, memory_ids: list[str]) -> int:
        raise NotImplementedError

    @abstractmethod
    async def oldest(
        self,
        *,
        limit: int = 100,
        include_archived: bool = False,
    ) -> list[MemoryRecord]:
        raise NotImplementedError

    @abstractmethod
    async def count(self, *, include_archived: bool = False) -> int:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Semantic vector search (optional capability).
    #
    # Repositories that support vector search override these. The default
    # implementations raise NotImplementedError so callers can feature-detect
    # via hasattr/try-except rather than guessing at config flags.
    # ------------------------------------------------------------------

    async def index_embedding(
        self,
        memory_id: str,
        embedding: list[float],
    ) -> None:
        """Persist an embedding vector for ``memory_id``.

        Implementations must be idempotent — re-indexing the same memory
        replaces the prior vector rather than duplicating it.
        """
        raise NotImplementedError("Repository does not support vector indexing")

    async def search_semantic(
        self,
        query_embedding: list[float],
        *,
        limit: int = 8,
        include_archived: bool = False,
    ) -> list[MemoryRecord]:
        """Return the top-K memories most similar to ``query_embedding``.

        Results are ordered by descending similarity. Repositories without
        vector support raise NotImplementedError.
        """
        raise NotImplementedError("Repository does not support vector search")

    async def count_embeddings(self) -> int:
        """Return the number of memories with a stored embedding vector."""
        raise NotImplementedError("Repository does not support vector indexing")
