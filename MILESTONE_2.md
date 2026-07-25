# Capsule Brain v2 — Milestone 2

Milestone 2 moves the architecture from "runtime primitives" to a real migrated
service path.

## Implemented

- `RedisBridge` using `redis.asyncio`
  - Redis is now optional cross-process transport.
  - Internal async services no longer execute on a Redis listener thread.
- `GoalPlannerV2`
  - real `CapsuleService`
  - SQLite persistence
  - stable goal/task IDs
  - async event handling
  - persisted task edits/status changes
- `runtime/bootstrap.py`
  - dependency-aware service construction
- `gui/app_runner.py`
  - one qasync event loop owns the entire process
  - top-level shutdown ownership
  - no nested GUI event loop
- v2 runtime configuration
- tests for real goal creation and ID-based task mutation

## Important integration boundary

The existing GUI must be changed from:

    gui_qt.launch(engine)

to a constructor/factory:

    window = create_window(runtime)

The GUI must **not** call `QEventLoop`, `asyncio.set_event_loop`, or
`run_forever()` itself.

## Next milestone

Milestone 3 should migrate `LLMAdapter` into `LLMGateway`, introduce model
capabilities/results, and replace the temporary deterministic goal decomposer
with structured LLM output.
