# Repository Guidelines

## Project Structure & Module Organization
`app/` contains the FastAPI entrypoint, API routes, core configuration, service orchestration, Celery tasks, and workers. `tests/` holds the pytest suite, with fixtures under `tests/fixtures/`. `scripts/` contains local demo, regression, and model-load smoke checks. `docs/` covers architecture and local deployment notes. `storage/` is used for local models, job state, and generated artifacts; avoid committing large runtime outputs.

## Build, Test, and Development Commands
Install base dev tooling with `pip install -e .[dev]`. Add ML dependencies only when needed with `pip install -e .[ml]`. Run the API locally with `uvicorn app.main:app --host 0.0.0.0 --port 8000`. Run tests with `pytest` or target a file such as `pytest tests/test_health.py`. Lint with `ruff check .`. Run the control regression script with `python -m scripts.run_control_regression`.

## Coding Style & Naming Conventions
Use Python 3.11+, 4-space indentation, and keep lines within the Ruff limit of 100 characters. Follow existing naming: `snake_case` for modules, functions, and test files; `PascalCase` for classes; clear service-oriented names in `app/services/`. Preserve type hints where the codebase already uses them, and prefer small, composable helpers over large mixed-responsibility functions.

## Testing Guidelines
Use `pytest` for all automated checks. Place new tests in `tests/` and name them `test_*.py`. Prefer targeted test runs before full-suite or regression runs. For model-loading validation, start with `python scripts/test_local_sdxl_load.py --skip-infer` or `python scripts/test_local_svd_load.py --skip-infer` before attempting real inference.

## Commit & Pull Request Guidelines
Recent history favors short, descriptive commits with optional prefixes such as `docs:`. Keep commits focused and explain the behavior change, not just the file touched. Pull requests should summarize user-visible changes, note any new environment variables or model requirements, and include logs or screenshots only when API or demo behavior changed.

## Configuration & Resource Notes
Prefer `VIDGEN_TASK_MODE=eager` and mock-safe paths for routine development. Do not blindly run heavy ML scripts or full inference on large inputs; use skip-infer smoke checks first and stop if memory pressure becomes unsafe. Keep local model assets under `storage/models/` and rely on `.env.example` when introducing new configuration.
