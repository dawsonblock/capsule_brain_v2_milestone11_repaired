# Capsule Brain v2 — Milestone 6

Milestone 6 creates the first complete operator-to-model-to-memory-to-response
path in the v2 architecture.

Implemented:

- Persistent ConversationRepository using SQLite/WAL.
- Stable conversation IDs.
- Stable user/assistant turn IDs.
- Stable assistant response IDs.
- Explicit parent-turn linkage.
- Exact model/provider provenance per response.
- LLM latency, attempt, usage, and route persistence.
- ConversationService as a managed CapsuleService.
- Conversation history context construction.
- Recent MemoryService retrieval injected into context.
- Operator and assistant turns written back into persistent memory.
- `conversation.message.create` input event.
- `conversation.response.created` canonical output event.
- `conversation.response.failed` failure event.
- `gui.response` compatibility event for GUI migration.
- End-to-end tests using a fake provider with no network/API key.

This is the first milestone where Capsule Brain v2 has a real chat pipeline:

operator
  -> conversation event
  -> persist user turn
  -> persist operator memory
  -> retrieve conversation + memory context
  -> LLMGateway route
  -> persist assistant turn
  -> persist exact response provenance
  -> persist assistant memory
  -> publish GUI response

Next milestone:

Milestone 7 should add FeedbackService + ExperienceStore so explicit operator
feedback is attached to exact response IDs instead of timestamps. After that,
ReflectionService can use verifier failures and unresolved goals as bounded
recursive-reasoning seeds.
