# Autonomous Knowledge Agent — working backup

Snapshot date: 2026-09-22. This is work in progress, not a final submission.

## Contents and origin

Copied from `NewWorldSolution/msc-ai`, directory
`curriculum/nd901-agentic-ai-langchain-langgraph/projects/autonomous-knowledge-agent/`.
The snapshot includes the current saved working files, including uncommitted changes.
The original study copy remains intact.

`BACKUP_MANIFEST.json` records the size and SHA-256 hash of each of the 80 copied
source files. Those files were copied without content changes. The manifest,
this status note, and the project-level `.gitignore` are backup metadata added
in this repository.

Included: implementation, executable tests, notebooks, architecture documentation
and diagram, requirements, course sample JSONL data, starter files, instructions,
rubric, environment-variable template, and temporary coaching guides.

Excluded: real `.env` credentials, local virtual environments, generated databases,
vector indexes, Python/test/editor caches, and operating-system metadata.
Repository-level learning records remain in the original study repository.
No secret-pattern or exact local credential-value matches were found in the copied
files during the pre-upload checks; this is not a guarantee of exhaustive secret detection.

## Verification

The copied `solution/` test suite ran from this location with the existing study
project's Python environment: **232 passed, 12 skipped, 3 warnings**.
All copied Python files passed syntax compilation. No dependencies were installed.
Tests used fake models and test database fixtures; live providers and a complete
end-to-end application run were not exercised. A fresh-environment installation
has not been validated.

Once the project dependencies and pytest are available, run from `solution/`:

```sh
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider
```

## Work still outstanding

- Knowledge, Response, Escalation, and the workflow factory still contain scaffolds.
- The application notebook and end-to-end scenarios are not finished.
- The external knowledge corpus currently contains four articles.
- Live-provider behavior and deployment-specific timeouts remain to be validated.
- Some documentation links refer to exercises in the original study repository;
  those exercises are not part of this project backup.

## Before final submission

Keep this backup for recovery. Prepare the submission separately using the original
project instructions. Exclude this status note and manifest, temporary coaching
reports such as `solution/TOOL_CORRECTIONS.md` and
`solution/FOUNDATION_FIXES_2026-09-17.md`, and other review-only material from the
submission package. Preserve implementation, useful docstrings, tests, and required
project artifacts. Do not delete study materials from this backup merely because
they are excluded from the eventual submission.

This is a snapshot, not an automatic synchronization setup. Later edits in the
study repository will require another deliberate backup update.
