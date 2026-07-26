from __future__ import annotations

from typing import Any

from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.events.models import EventEnvelope
from capsule_brain.runtime.service import CapsuleService, HealthStatus, ServiceState

from .embeddings import EmbeddingProvider, HashEmbeddingProvider, NullEmbeddingProvider
from .models import MemoryRecord, MemoryType
from .sqlite_repository import SQLiteMemoryRepository


PROTECTED_TYPES = {
    MemoryType.OPERATOR,
    MemoryType.GOAL,
    MemoryType.FEEDBACK,
    MemoryType.SYSTEM,
}


class MemoryService(CapsuleService):
    name = "memory"

    def __init__(
        self,
        event_bus: LocalEventBus,
        cfg: dict[str, Any] | None = None,
        repository: SQLiteMemoryRepository | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        super().__init__(cfg)
        self.event_bus = event_bus
        self.repository = repository or SQLiteMemoryRepository(
            self.cfg.get("db_path", "data/memory_v2.sqlite")
        )
        # Embedding provider for semantic search. Defaults to a deterministic
        # hash-based embedding so the system has a working vector pipeline
        # with zero external dependencies. Configure a real embedding model
        # (e.g. OpenAI text-embedding) via cfg["embedding"]["provider"] for
        # production-quality semantic retrieval.
        self.embedding_provider: EmbeddingProvider = (
            embedding_provider or self._build_embedding_provider()
        )
        # When True, write() automatically indexes an embedding for every new
        # memory so search_semantic has data to work with. Disable to index
        # explicitly via index_memory().
        self.auto_index = bool(self.cfg.get("auto_index_embeddings", True))

    def _build_embedding_provider(self) -> EmbeddingProvider:
        emb_cfg = self.cfg.get("embedding", {}) or {}
        kind = str(emb_cfg.get("provider", "hash")).lower()
        if kind in {"none", "null", "disabled"}:
            return NullEmbeddingProvider()
        if kind == "hash":
            return HashEmbeddingProvider(
                dimension=int(emb_cfg.get("dimension", 128))
            )
        # Unknown providers fall back to hash embedding rather than failing —
        # memory is a core service and must always start.
        return HashEmbeddingProvider(
            dimension=int(emb_cfg.get("dimension", 128))
        )

    async def start(self) -> None:
        self.state = ServiceState.STARTING
        await self.repository.start()
        self.state = ServiceState.RUNNING

    async def stop(self) -> None:
        self.state = ServiceState.STOPPING
        await self.repository.stop()
        self.state = ServiceState.STOPPED

    async def write(
        self,
        text: str,
        *,
        type: MemoryType = MemoryType.OBSERVATION,
        source: str = "unknown",
        tags: list[str] | None = None,
        conversation_id: str | None = None,
        turn_id: str | None = None,
        importance: float = 0.0,
        confidence: float | None = None,
        protected: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        record = MemoryRecord(
            text=text,
            type=type,
            source=source,
            tags=list(tags or []),
            conversation_id=conversation_id,
            turn_id=turn_id,
            importance=float(importance),
            confidence=confidence,
            protected=type in PROTECTED_TYPES if protected is None else protected,
            metadata=dict(metadata or {}),
        )
        await self.repository.create(record)

        # Auto-index an embedding so semantic search has data to work with.
        # Failures here are non-fatal: semantic search is a quality upgrade,
        # not a correctness requirement, and a transient embedding provider
        # outage must not break memory writes.
        if self.auto_index and self.embedding_provider.dimension > 0:
            try:
                embedding = await self.embedding_provider.embed(text)
                if embedding:
                    await self.repository.index_embedding(record.id, embedding)
            except Exception:
                pass

        await self.event_bus.publish(
            EventEnvelope(
                event_type="memory.created",
                source=self.name,
                payload={
                    "id": record.id,
                    "type": record.type.value,
                    "source": record.source,
                    "protected": record.protected,
                },
            )
        )
        return record

    async def recent(
        self,
        *,
        limit: int = 100,
        include_archived: bool = False,
    ) -> list[MemoryRecord]:
        return await self.repository.recent(
            limit=limit,
            include_archived=include_archived,
        )

    async def oldest(
        self,
        *,
        limit: int = 100,
        include_archived: bool = False,
    ) -> list[MemoryRecord]:
        return await self.repository.oldest(
            limit=limit,
            include_archived=include_archived,
        )

    async def search_semantic(
        self,
        query: str,
        *,
        limit: int = 8,
        include_archived: bool = False,
    ) -> list[MemoryRecord]:
        """Return the top-K memories semantically most similar to ``query``.

        The query is embedded using the configured EmbeddingProvider and the
        resulting vector is passed to the repository's vector search. When no
        embedding provider is configured (NullEmbeddingProvider), this raises
        a clear error rather than silently returning chronological results —
        silent fallback would mask a misconfiguration.
        """
        if self.embedding_provider.dimension == 0:
            raise RuntimeError(
                "Semantic search is disabled: no embedding provider configured"
            )
        embedding = await self.embedding_provider.embed(query)
        if not embedding:
            return []
        return await self.repository.search_semantic(
            embedding,
            limit=limit,
            include_archived=include_archived,
        )

    async def index_memory(self, memory_id: str) -> bool:
        """(Re)index an existing memory's embedding.

        Useful when auto-indexing was disabled at write time, or when an
        embedding provider is upgraded and existing memories need to be
        re-embedded. Returns True if the memory was found and indexed.
        """
        record = await self.repository.get(memory_id)
        if record is None or self.embedding_provider.dimension == 0:
            return False
        embedding = await self.embedding_provider.embed(record.text)
        if not embedding:
            return False
        await self.repository.index_embedding(memory_id, embedding)
        return True

    async def archive(self, memory_id: str) -> bool:
        archived = await self.repository.archive(memory_id)
        if archived:
            await self.event_bus.publish(
                EventEnvelope(
                    event_type="memory.archived",
                    source=self.name,
                    payload={"id": memory_id},
                )
            )
        return archived

    async def archive_batch(self, memory_ids: list[str]) -> int:
        """Archive a batch of memories in a single transaction.

        Returns the number of records actually archived (protected records
        are skipped). Publishes a single ``memory.archived_batch`` event.
        """
        count = await self.repository.archive_batch(memory_ids)
        if count > 0:
            await self.event_bus.publish(
                EventEnvelope(
                    event_type="memory.archived_batch",
                    source=self.name,
                    payload={"count": count, "requested_ids": list(memory_ids)},
                )
            )
        return count

    async def health(self) -> HealthStatus:
        embedding_count = 0
        vec_available = False
        if self.state in {ServiceState.RUNNING, ServiceState.DEGRADED}:
            try:
                embedding_count = await self.repository.count_embeddings()
            except NotImplementedError:
                embedding_count = 0
            vec_available = getattr(self.repository, "_vec_available", False)
        return HealthStatus(
            state=self.state,
            details={
                "active_count": await self.repository.count()
                if self.state in {ServiceState.RUNNING, ServiceState.DEGRADED}
                else 0,
                "total_count": await self.repository.count(include_archived=True)
                if self.state in {ServiceState.RUNNING, ServiceState.DEGRADED}
                else 0,
                "db_path": str(self.repository.db_path),
                "embedding_provider": self.embedding_provider.name,
                "embedding_dimension": self.embedding_provider.dimension,
                "embedding_count": embedding_count,
                "vec_extension": vec_available,
                "auto_index": self.auto_index,
            },
        )
