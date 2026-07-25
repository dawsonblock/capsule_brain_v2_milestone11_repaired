# Capsule Brain v2 — Milestone 10
Adds policy-gated execution-backed verification primitives.

Implemented:
- ExecutionPolicy / ExecutionRequest / ExecutionResult
- command allowlist
- cwd jail
- subprocess timeout
- output caps
- SQLite/WAL execution provenance
- execution.request / completed / failed
- runner and policy tests

Execution is disabled by default.

Important: this is restricted host execution, not a hardened sandbox.
Milestone 11 should move execution into a container or microVM worker with
network isolation, non-root execution, CPU/RAM/process limits, and disposable storage.
