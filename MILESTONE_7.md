# Capsule Brain v2 — Milestone 7

Milestone 7 replaces timestamp-based pseudo-reinforcement with exact-response
feedback attribution and a persistent experience dataset.

Implemented:

- `FeedbackRecord` with exact `response_id`.
- Feedback classes: positive, negative, neutral, correction.
- Persistent `ExperienceStore` using SQLite/WAL.
- Experience records include:
  - response ID
  - conversation ID
  - user/assistant turn IDs
  - prompt/response text
  - model/provider provenance
  - route
  - latency
  - attempts
  - usage
  - verifier placeholder
  - feedback fields
- `FeedbackService` as a managed service.
- `feedback.submit`, `feedback.recorded`, and `feedback.failed` events.
- Feedback is written to protected memory.
- No timestamp heuristics.
- No automatic model-weight or policy updates yet.

This creates the first legitimate learning substrate:

interaction
  -> exact response
  -> outcome / feedback
  -> persistent experience record
  -> later offline optimization

Next milestone:

Milestone 8 should wire ConversationService to automatically create an
ExperienceRecord for every assistant response and then build ReflectionService
with bounded iterative refinement seeded by verifier failures, unresolved goals,
or explicit operator reflection requests.
