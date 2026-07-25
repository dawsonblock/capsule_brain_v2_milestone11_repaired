# Capsule Brain v2 — Milestone 1

This milestone replaces the most dangerous part of the original architecture:
unowned asyncio tasks and thread-bound callbacks calling async functions.

Implemented:

1. `CapsuleService` lifecycle contract.
2. Dependency-aware `ServiceRegistry`.
3. `TaskRegistry` for owned background coroutines.
4. Async-safe `LocalEventBus`.
5. Top-level `CapsuleApplication`.
6. Unit tests for lifecycle, task cancellation, and sync/async event dispatch.

## Integration order

Copy `src/capsule_brain/runtime/` and `src/capsule_brain/events/` into the
existing project first.

Do **not** convert all services immediately.

Next migration target:

1. Replace `RedisBus` as the internal event mechanism with `LocalEventBus`.
2. Keep Redis only as an optional external-process bridge.
3. Convert `LLMAdapter` into a real `CapsuleService`.
4. Convert `GoalPlanner` next because it demonstrates the current
   Redis-thread → `asyncio.create_task()` failure.
5. Replace the GUI-owned event loop with one qasync loop owned by the
   application entry point.

## Current boundary

This milestone intentionally does not modify the legacy modules. It provides
the runtime they should be migrated onto.

That separation is deliberate: mixing a runtime rewrite with cognitive
behavior changes makes failures much harder to diagnose.
