# Capsule Brain v2 — Milestone 8

Milestone 8 replaces the old pseudo-recursive loop with a real bounded
ReflectionService.

Implemented:

- Persistent ReflectionSession records in SQLite/WAL.
- Iterative cycle:
  seed -> critique -> revision -> evaluation -> next iteration.
- Iteration N uses revision N-1.
- Explicit maximum iteration budget.
- Resolution stop condition.
- Duplicate-revision stop condition.
- Persistent final reflection memory.
- Reflection triggers:
  - explicit `reflection.request`
  - negative/correction feedback
  - `verification.failed`
  - `goal.unresolved`
- Events:
  - `reflection.completed`
  - `reflection.failed`
- Health metrics for runs/failures/session count.

This is materially different from the old recursive loop, which incremented a
depth counter but generated unrelated random seeds.

Next milestone:

Milestone 9 should introduce a verifier framework and route failures into this
ReflectionService, starting with code/schema/consistency verifiers and a
verification result model.
