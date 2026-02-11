# Repository Guidelines

This repository hosts automation agents and shared tooling. Follow the practices below to keep contributions consistent.

## Project Structure & Module Organization
- Store production agent logic under `agents/` with one subfolder per persona (for example, `agents/researcher/`).
- Shared libraries live in `shared/`; keep cross-cutting utilities in `shared/utils/`.
- Place scripts for local workflows inside `scripts/`; ensure any executable script has a `.ps1` or `.sh` mate depending on platform support.
- Keep tests in `tests/` mirroring the module tree, and documentation assets in `docs/`.

## Build, Test, and Development Commands
- `python -m venv .venv` then `pip install -r requirements.txt` to provision dependencies.
- `invoke lint` runs formatting and static analysis; add new checks to `tasks.py`.
- `pytest` executes the automated test suite.
- `python -m agents.cli` runs the default agent orchestrator for manual smoke tests.

## Coding Style & Naming Conventions
- Target Python 3.11; use 4-space indentation and type hints on all public functions.
- Follow Black and isort defaults; run `invoke lint` before pushing.
- Name agent entrypoints `run_<role>.py` and exported classes in PascalCase; internal helpers stay snake_case.
- Configuration files use lowercase kebab-case (for example, `config/default-runtime.yaml`).

## Testing Guidelines
- Write Pytest modules that mirror source paths (e.g., `agents/researcher/run_researcher.py` -> `tests/agents/researcher/test_run_researcher.py`).
- Require tests for new behaviours plus regression coverage for bugs.
- Use `pytest -m "not slow"` for routine CI and reserve slow tags for long-running flows.
- Capture sample transcripts under `tests/fixtures/` to document expected agent-tool exchanges.

## Commit & Pull Request Guidelines
- Use Conventional Commits (`feat:`, `fix:`, `docs:`, etc.) with concise, imperative subjects.
- Squash work-in-progress commits before opening a PR.
- PR descriptions must include: purpose summary, testing evidence (`invoke lint && pytest` output), and any follow-up tasks.
- Link relevant issues and attach screenshots or logs when behaviour changes.

## Security & Configuration Tips
- Never commit API keys; store them in `.env.local` and reference via `dotenv`.
- Review `configs/secrets.example.yaml` before adding new integrations and document required environment variables in `docs/configuration.md`.

