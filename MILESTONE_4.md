# Capsule Brain v2 — Milestone 4

Milestone 4 upgrades the LLM layer from a typed adapter into a real routing and
resilience boundary.

Implemented:

- Deterministic route plans with ordered model fallback.
- Per-model circuit breakers.
- Provider health accounting.
- Global bounded concurrency/backpressure.
- Cancellation propagation.
- Request tracing.
- Token accounting.
- Configurable estimated model cost accounting.
- Streaming interface with safe fallback semantics.
- Fallback-aware usage metrics.
- Gateway health reports active requests, providers, breakers, usage, and trace data.
- New tests for:
  - model fallback
  - circuit breaking
  - concurrency limits
  - cancellation
  - streaming
  - usage/cost accounting

Important design rule:

Fallback is allowed before output is committed. For streaming, once the first
chunk has been emitted, the gateway does not silently switch models because that
would splice outputs from different models into one response.

Next milestone:

Milestone 5 should build the persistent MemoryRepository, UUID-based memory
schema, transactional SQLite/WAL storage, archive-first consolidation, and then
migrate CapsuleEngine away from direct `_memory` access.
