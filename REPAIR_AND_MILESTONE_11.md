# Capsule Brain v2 — Repair Release + Milestone 11

This release applies the full DEF-01 through DEF-07 repair set and adds a
container-backed execution runner behind a disabled-by-default policy gate.

Repairs:
- ExperienceStore promoted to managed CapsuleService.
- ExperienceStore registered in ServiceRegistry.
- ExecutionRepository shutdown serialized with repository lock.
- Execution command policy normalized for Windows extensions.
- Caller-controlled executable paths rejected.
- Execution failure metrics include non-zero exits and timeouts.
- FeedbackService uses ConversationRepository as fallback.
- Reflection duplicate detection starts with normalized seed.
- execution and verification modules expanded to readable PEP 8 structure.

Milestone 11:
- ContainerExecutionRunner.
- Docker/Podman selectable engine.
- network disabled.
- non-root user.
- memory and CPU caps.
- PID limit.
- read-only root filesystem.
- all Linux capabilities dropped.
- no-new-privileges.
- read-only workspace mount.
- disposable container (--rm).

Execution remains disabled by default. Container-backed mode must be explicitly
enabled in configuration.
