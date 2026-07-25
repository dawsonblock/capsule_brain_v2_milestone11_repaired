# Capsule Brain v2 — Milestone 3

Milestone 3 replaces the direct provider-facing LLM adapter pattern with a typed
gateway boundary.

Implemented:

- `LLMGateway` as a managed CapsuleService.
- Typed `LLMRequest` / `LLMResult`.
- Explicit model capability declarations.
- Provider isolation behind `LLMProvider`.
- OpenAI-compatible async provider using `httpx.AsyncClient`.
- Central timeout handling.
- Bounded retry with exponential backoff.
- Explicit configuration, capability, timeout, and provider errors.
- JSON structured-output helper.
- `StructuredGoalDecomposer` for Goal Planner integration.
- Unit tests using a deterministic fake provider; tests require no network/API key.

Architectural rule:

Core services depend on `LLMGateway`, never on OpenAI/httpx/provider-specific
payloads. Provider details stop at `llm/providers/`.

Next:

Milestone 4 should add provider routing/fallback, concurrency limits,
circuit-breakers, request tracing, token/cost accounting, cancellation-aware
streaming, and full bootstrap integration of the structured goal decomposer.
