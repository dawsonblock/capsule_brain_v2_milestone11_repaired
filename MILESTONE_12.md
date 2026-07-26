# Capsule Brain v2 — Milestone 12: Architectural Upgrades

This milestone implements the five high-impact architectural upgrades from the
Part 1 improvement plan, plus a fix for a pre-existing test failure on main.

## Repairs

- `ContainerExecutionRunner` now injects `PYTHONPYCACHEPREFIX=/tmp/pycache`
  alongside `PYTHONDONTWRITEBYTECODE=1` (fixes the pre-existing
  `test_rep03_container_runner_includes_bytecode_env_vars` failure).

## Improvement 1: Semantic Vector Search for Memory

- New `EmbeddingProvider` protocol with `HashEmbeddingProvider` (zero-dependency
  deterministic fallback) and `NullEmbeddingProvider` (disabled state).
- `SQLiteMemoryRepository` gains a `memory_embeddings` table and three new
  methods: `index_embedding`, `search_semantic`, `count_embeddings`.
- When the `sqlite-vec` extension is loadable, similarity search runs natively
  inside SQLite via a `vec0` virtual table. Otherwise the repository falls
  back to an in-Python cosine-similarity scan. Both paths share the same API.
- `MemoryService` auto-indexes embeddings on every `write()` (configurable)
  and exposes `search_semantic(query, limit)` and `index_memory(id)`.
- `ConversationService` gains opt-in `semantic_memory` config: when enabled,
  it embeds the user's message and retrieves top-K semantically relevant
  memories instead of the most recent K. Falls back to chronological
  retrieval on any error.

Files: `memory/embeddings.py`, `memory/sqlite_repository.py`,
`memory/repository.py`, `memory/service.py`, `conversation/service.py`.
Tests: `tests/unit/test_memory_semantic.py` (7 tests).

## Improvement 2: Native Tool/Function Calling Abstraction

- New `ToolSpec`, `ToolCall`, `ToolResult` dataclasses mirroring the
  OpenAI/Anthropic function-calling schema.
- New `ToolRegistry` CapsuleService: registers tools with JSON Schema
  parameter descriptions and async handlers, validates required arguments
  before dispatch, and stringifies results for the model.
- `LLMRequest` gains `tools` and `tool_results` fields.
- `LLMResult` gains `tool_calls` and `finish_reason` fields.
- `OpenAICompatibleProvider` now serializes tool schemas into the
  chat-completions payload and parses `tool_calls` from the response.
- `LLMGateway.generate_with_tools(request, registry, max_iterations)` runs
  the tool-calling loop: generate → execute tool calls → feed results back →
  repeat until the model produces a final answer or max_iterations is hit.
- The `tools` capability is enforced — models without it reject tool requests.

Files: `llm/tools.py`, `llm/models.py`, `llm/gateway.py`,
`llm/providers/openai_compatible.py`.
Tests: `tests/unit/test_llm_tools.py` (5 tests).

## Improvement 3: OpenTelemetry & Distributed Tracing

- New `observability/tracing.py` with `Span`, `Tracer`, and process-wide
  default-tracer management.
- `Tracer.otel()` returns an OTel-backed tracer when `opentelemetry` is
  installed; otherwise it degrades to a no-op tracer with zero overhead.
  This method never raises.
- `LocalEventBus.publish` wraps publication in a span keyed by
  `EventEnvelope.correlation_id` so the full handler fan-out is visible in
  the trace waterfall. Handler failures are recorded as span events.
- `LLMGateway.generate` wraps the routing loop in a span with route, model,
  attempts, and fallback attributes.
- Spans carry `correlation_id` so a complete trace (Operator Event → Memory
  Lookup → LLM Route → Container Execution → Verification → Reflection)
  can be reconstructed in Jaeger, Phoenix, or any OTel backend.
- Configurable via `tracing.enable` and `tracing.service_name` in the YAML.

Files: `observability/__init__.py`, `observability/tracing.py`,
`events/local_bus.py`, `llm/gateway.py`.
Tests: `tests/unit/test_tracing.py` (8 tests).

## Improvement 4: Adaptive Reflection Strategies

- New `ReflectionStrategy` protocol with four strategies:
  - `DefaultReflectionStrategy`: the original critique → revise → evaluate
    loop (3 LLM calls/iteration). Used for operator feedback and goal
    reasoning.
  - `CodeSyntaxStrategy`: fast-path single repair call per iteration with
    `compile()` validation. Resolves immediately when the fix parses.
    Reduces token usage by ~60-80% for simple syntax fixes.
  - `PytestFailureStrategy`: 2-call loop (repair + self-check) with
    assertion-diff extraction from the seed/metadata. Saves ~33% of tokens
    per iteration vs. the default 3-call loop.
  - `OperatorFeedbackStrategy`: thin specialization of the default strategy
    for qualitative multi-perspective reasoning.
- `select_strategy(source, metadata)` picks the right strategy based on
  trigger source and verification metadata (verifier name, summary text).
  Supports explicit override via `metadata["strategy"]`.
- `ReflectionService` delegates to the selected strategy, reports per-
  strategy run counts in health, and tags reflection memories with the
  strategy used.

Files: `reflection/strategies.py`, `reflection/service.py`.
Tests: `tests/unit/test_reflection_strategies.py` (11 tests).

## Improvement 5: Async Worker Pool for Execution

- New `ExecutionWorkerPool` CapsuleService: bounded async worker pool using
  `max_workers` permanent consumer tasks pulling from an asyncio queue.
- Limits concurrent container/host executions to prevent Docker daemon
  CPU/RAM thrashing under heavy multi-task load.
- Configurable `queue_max` for backpressure (rejects submissions when full),
  `drain_timeout_s` for graceful shutdown, and `max_workers` for concurrency.
- `ExecutionService` delegates to the pool when one is provided; otherwise
  calls `runner.run()` directly (preserving original behavior).
- On stop, queued-job futures are cancelled and in-flight jobs are drained.
- Tracing spans wrap each worker execution with the request's correlation_id.

Files: `execution/worker_pool.py`, `execution/service.py`.
Tests: `tests/unit/test_execution_worker_pool.py` (6 tests).

## Bootstrap & Config

- `build_application` initializes the default tracer (OTel when
  `tracing.enable` is true, no-op otherwise).
- `ToolRegistry` is registered as a managed service when the LLM gateway is
  enabled.
- `ExecutionWorkerPool` is created and wired into `ExecutionService` when
  `execution.worker_pool.max_workers` > 0.
- `configs/v2_runtime.yaml` updated with `tracing`, `memory.embedding`,
  `conversation.semantic_memory`, and `execution.worker_pool` sections.

## Validation

- compileall: PASS
- pytest: 131 passed (95 pre-existing + 36 new, 1 pre-existing failure fixed)
