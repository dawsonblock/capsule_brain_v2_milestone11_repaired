from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Any


class EmbeddingProvider(ABC):
    """Pluggable text embedding provider.

    Implementations may call a local model, an external embedding API, or a
    deterministic hash-based stub for testing. The contract is intentionally
    minimal: text in, fixed-length float vector out.
    """

    name: str = "embedding"
    dimension: int = 0

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError


class HashEmbeddingProvider(EmbeddingProvider):
    """Deterministic hash-based embedding provider.

    This is a zero-dependency fallback used when no real embedding model is
    configured. It produces a fixed-dimensional vector by hashing token n-grams
    and projecting them into buckets. It is NOT semantically meaningful — its
    only purpose is to give the vector store a working embedding pipeline for
    tests and bootstrapping, and to make similarity search deterministic so
    identical/near-identical strings still cluster together.

    For production semantic retrieval, configure a real embedding provider
    (e.g. an OpenAI text-embedding model) via the memory service config.
    """

    name = "hash"
    dimension = 128

    def __init__(self, *, dimension: int = 128) -> None:
        self.dimension = max(1, int(dimension))

    async def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        normalized = text.lower().strip()
        if not normalized:
            return vec
        # Tokenize on whitespace and punctuation boundaries. Keep it simple —
        # the goal is reproducibility, not linguistic accuracy.
        tokens = [t for t in normalized.split() if t]
        # Unigrams + bigrams so shared phrases increase overlap.
        grams: list[str] = list(tokens)
        for i in range(len(tokens) - 1):
            grams.append(f"{tokens[i]} {tokens[i + 1]}")
        for gram in grams:
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[bucket] += sign
        # L2 normalize so cosine similarity reduces to dot product.
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0.0:
            vec = [v / norm for v in vec]
        return vec


class NullEmbeddingProvider(EmbeddingProvider):
    """No-op embedding provider used when semantic search is disabled.

    Returns an empty vector. The repository checks the dimension to detect
    this and short-circuits semantic queries with a clear error rather than
    silently returning chronological results.
    """

    name = "null"
    dimension = 0

    async def embed(self, text: str) -> list[float]:
        return []


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity for two equal-length float vectors.

    Vectors are assumed to be pre-normalized (HashEmbeddingProvider and most
    real embedding APIs return unit vectors), but we defend against
    non-normalized inputs by computing the full cosine formula.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def serialize_vector(vec: list[float]) -> str:
    """Serialize a float vector to a compact, JSON-compatible string.

    Stored as a comma-separated string of floats rather than a JSON array to
    keep the row size small and parsing fast for the in-Python fallback path.
    """
    return ",".join(f"{x:.6f}" for x in vec)


def deserialize_vector(blob: str) -> list[float]:
    if not blob:
        return []
    return [float(part) for part in blob.split(",") if part]


def try_load_sqlite_vec(conn: Any) -> bool:
    """Attempt to load the sqlite-vec extension into ``conn``.

    Returns True if the extension loaded successfully, False otherwise. The
    caller is expected to fall back to the in-Python similarity path when this
    returns False. We swallow all errors here because extension availability
    is environment-dependent (platform, Python build, wheel install) and must
    never be a hard failure for the memory subsystem.
    """
    try:
        import sqlite_vec  # type: ignore[import-not-found]
    except Exception:
        return False
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return True
    except Exception:
        try:
            conn.enable_load_extension(False)
        except Exception:
            pass
        return False
