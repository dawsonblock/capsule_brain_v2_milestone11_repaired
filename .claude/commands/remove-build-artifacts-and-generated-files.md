---
name: remove-build-artifacts-and-generated-files
description: Workflow command scaffold for remove-build-artifacts-and-generated-files in capsule_brain_v2_milestone11_repaired.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /remove-build-artifacts-and-generated-files

Use this workflow when working on **remove-build-artifacts-and-generated-files** in `capsule_brain_v2_milestone11_repaired`.

## Goal

Removes compiled Python bytecode (.pyc) and database files (.sqlite, .sqlite-shm, .sqlite-wal) from git tracking after .gitignore is updated.

## Common Files

- `src/**/__pycache__/*.pyc`
- `data/*.sqlite*`
- `tests/unit/__pycache__/*.pyc`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Identify .pyc and .sqlite* files that are tracked by git.
- Remove these files from version control.
- Ensure .gitignore covers these patterns.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.