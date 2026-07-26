import pytest

from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.memory.embeddings import (
    HashEmbeddingProvider,
    NullEmbeddingProvider,
    cosine_similarity,
    deserialize_vector,
    serialize_vector,
)
from capsule_brain.memory.models import MemoryType
from capsule_brain.memory.service import MemoryService
from capsule_brain.memory.sqlite_repository import SQLiteMemoryRepository


@pytest.mark.asyncio
async def test_hash_embedding_is_deterministic_and_normalized():
    provider = HashEmbeddingProvider(dimension=32)
    a = await provider.embed("hello world foo")
    b = await provider.embed("hello world foo")
    c = await provider.embed("completely different text")
    assert a == b
    assert len(a) == 32
    # L2-normalized
    norm = sum(x * x for x in a) ** 0.5
    assert abs(norm - 1.0) < 1e-6
    # Identical text is perfectly similar
    assert cosine_similarity(a, b) == pytest.approx(1.0)
    # Different text is less similar than identical text
    assert cosine_similarity(a, c) < 1.0


@pytest.mark.asyncio
async def test_null_embedding_returns_empty():
    provider = NullEmbeddingProvider()
    assert provider.dimension == 0
    assert await provider.embed("anything") == []


def test_serialize_deserialize_roundtrip():
    vec = [0.1, -0.2, 0.3, 0.0]
    blob = serialize_vector(vec)
    assert deserialize_vector(blob) == pytest.approx(vec)


@pytest.mark.asyncio
async def test_semantic_search_retrieves_relevant_memories(tmp_path):
    """Memories sharing tokens with the query should rank above unrelated ones."""
    bus = LocalEventBus()
    await bus.start()
    repo = SQLiteMemoryRepository(str(tmp_path / "memory.sqlite"))
    # Use a high dimension so hash-bucket collisions do not drown out signal.
    memory = MemoryService(
        bus,
        repository=repo,
        cfg={"embedding": {"dimension": 512}},
    )
    await memory.start()

    await memory.write(
        "The user prefers Python over JavaScript for scripting tasks.",
        type=MemoryType.OBSERVATION,
        source="test",
    )
    await memory.write(
        "Reminder: take out the trash on Tuesdays.",
        type=MemoryType.OBSERVATION,
        source="test",
    )
    await memory.write(
        "Python is the configured default interpreter for the sandbox.",
        type=MemoryType.OBSERVATION,
        source="test",
    )

    results = await memory.search_semantic("python user preference", limit=3)
    assert len(results) == 3
    # The two Python-related memories should rank above the trash reminder.
    top_texts = " ".join(r.text for r in results[:2]).lower()
    assert "python" in top_texts
    assert "trash" not in top_texts

    await memory.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_semantic_search_disabled_without_embedding_provider(tmp_path):
    bus = LocalEventBus()
    await bus.start()
    repo = SQLiteMemoryRepository(str(tmp_path / "memory.sqlite"))
    memory = MemoryService(
        bus,
        repository=repo,
        cfg={"embedding": {"provider": "none"}},
    )
    await memory.start()

    with pytest.raises(RuntimeError, match="no embedding provider"):
        await memory.search_semantic("query")

    await memory.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_index_memory_reindexes_existing_record(tmp_path):
    bus = LocalEventBus()
    await bus.start()
    repo = SQLiteMemoryRepository(str(tmp_path / "memory.sqlite"))
    memory = MemoryService(
        bus,
        repository=repo,
        cfg={"auto_index_embeddings": False},
    )
    await memory.start()

    record = await memory.write("hello world", source="test")
    # No embedding indexed yet because auto_index is off.
    assert await memory.repository.count_embeddings() == 0
    assert await memory.index_memory(record.id) is True
    assert await memory.repository.count_embeddings() == 1
    # Idempotent re-index
    assert await memory.index_memory(record.id) is True
    assert await memory.repository.count_embeddings() == 1

    await memory.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_health_reports_embedding_status(tmp_path):
    bus = LocalEventBus()
    await bus.start()
    repo = SQLiteMemoryRepository(str(tmp_path / "memory.sqlite"))
    memory = MemoryService(bus, repository=repo)
    await memory.start()
    await memory.write("hello", source="test")

    health = await memory.health()
    assert health.details["embedding_provider"] == "hash"
    assert health.details["embedding_dimension"] > 0
    assert health.details["embedding_count"] == 1

    await memory.stop()
    await bus.stop()
