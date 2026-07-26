---
name: add-new-core-service-with-persistence-and-tests
description: Workflow command scaffold for add-new-core-service-with-persistence-and-tests in capsule_brain_v2_milestone11_repaired.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /add-new-core-service-with-persistence-and-tests

Use this workflow when working on **add-new-core-service-with-persistence-and-tests** in `capsule_brain_v2_milestone11_repaired`.

## Goal

Implements a new core service or architectural upgrade, including models, repository, service logic, configuration, and corresponding unit tests.

## Common Files

- `src/capsule_brain/<feature>/{models.py,repository.py,service.py,runner.py,__init__.py}`
- `src/capsule_brain/runtime/bootstrap.py`
- `configs/v2_runtime.yaml`
- `tests/unit/test_<feature>*.py`
- `MILESTONE_12.md`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Define or update models in a new or existing models.py file.
- Implement repository logic in repository.py (with SQLite or other storage).
- Implement service logic in service.py or runner.py.
- Wire up the new service in runtime/bootstrap.py and update configs/v2_runtime.yaml as needed.
- Add or update __init__.py and supporting files in the new/existing module.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.