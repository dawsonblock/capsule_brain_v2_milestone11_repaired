# Capsule Brain v2 — Milestone 5

Milestone 5 replaces mutable in-process memory with a transactional persistent
memory subsystem.

Implemented:

- UUID-backed `MemoryRecord`.
- Explicit memory types.
- Protected-memory policy.
- SQLite persistence with WAL mode.
- Transactional create/update/archive operations.
- Async serialization around sqlite3 connection access.
- `MemoryService` as the only public memory mutation boundary.
- `memory.created` / `memory.archived` events.
- Archive-first `MemoryConsolidator`.
- No physical deletion.
- Fail-closed consolidation behavior.
- `CapsuleEngineV2` facade with no public `_memory` list.
- Concurrency tests for 200 simultaneous writes.
- Protected records cannot be automatically archived.

Next milestone:

Milestone 6 should build ConversationService on top of MemoryService and
LLMGateway, add conversation/turn/response IDs, persist exact model provenance,
publish GUI response events, and create the first real end-to-end chat path.
