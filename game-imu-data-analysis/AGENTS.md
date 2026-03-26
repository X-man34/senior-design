# AGENTS.md

This repository is primarily maintained by AI coding agents. The agent is expected to take ownership of routine development work and keep the project organized, reliable, and easy for humans to review.

## Core Responsibilities

- Use sound software engineering judgment and prefer simple, maintainable solutions.
- Keep the repository in a healthy state at all times.
- Make steady progress without waiting for unnecessary human direction.
- Leave clear commit history and readable code for future contributors.

## Engineering Best Practices

- Understand the existing code and project structure before making major changes.
- Always use the repository's existing local virtual environment for Python commands and package access. Do not create a new virtual environment if a local one already exists.
- Prefer small, composable modules over large monolithic scripts.
- Add or update tests when behavior changes.
- Validate work with the relevant test, lint, or smoke-check commands before finishing.
- Document important assumptions, data formats, and workflow changes.
- Avoid introducing unnecessary dependencies or overly complex abstractions.
- Preserve raw data integrity and avoid mutating original datasets in place.

## Repo Management

- Keep source code in `src/`, runnable helpers in `scripts/`, tests in `tests/`, and datasets under `data/`.
- Do not commit large raw datasets, generated artifacts, secrets, or local environment files.
- Update `.gitignore` when new generated outputs or local-only files appear.
- Keep the README and setup instructions current as the project evolves.
- Prefer reproducible workflows and deterministic scripts where practical.

## Git Workflow

- Commit frequently in small, logical increments.
- Use clear commit messages that explain the purpose of the change.
- Before committing, review the diff and confirm the change is scoped correctly.
- Do not rewrite or discard user-authored work unless explicitly instructed.
- If the worktree contains unrelated changes, avoid interfering with them.
- When completing a meaningful milestone, create a commit instead of leaving large uncommitted changes.

## Working Style

- Be proactive: identify the next useful improvement and make progress.
- Be cautious with destructive actions and prefer reversible changes.
- Surface risks, assumptions, or blockers clearly when they matter.
- Favor code and documentation that a student team can understand and extend.
