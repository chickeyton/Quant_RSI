# Repository Guidelines

## Project Structure & Module Organization

Quant RSI is a Python quantitative research and forecast system. Shared code lives in `src/quant_rsi/`, organized into `core/`, `market/`, `forecast/`, `research/`, `evaluation/`, and `execution/`. Instrument profiles live in `instruments/` and operating/design documentation in `docs/`. Preserve the reference document `2505.15155v2.pdf`.

Place source code in `src/`, automated tests in `tests/`, and runnable examples in `examples/`. Keep [README.md](README.md) synchronized with the project purpose, prerequisites, and directory layout. Create directories only when they contain useful files.

## Build, Test, and Development Commands

Use Linux and Python 3.12, as validated by CI. Follow installation instructions in [README.md](README.md). Dependencies are pinned in `requirements-dev.lock`; package and tool configuration live in `pyproject.toml`. From the activated environment, run:

```bash
ruff check src tests examples/profile_validation.py
ruff format --check src tests examples/profile_validation.py
pytest -q
```

Start with tests relevant to the change, then run the required broader checks. CI runs these checks in [.github/workflows/ci.yml](.github/workflows/ci.yml).

For measured runs, `python examples/profile_validation.py --output /tmp/quant-rsi-validation`
executes the complete suite and saves commands, environment, per-test timings and
exit codes. Use a fresh output directory. `--shards 2` preserves module fixtures
when running in parallel, including CI; focused selectors are development feedback,
not a substitute for the complete required suite. See [validation cost](docs/validation_cost.md).
Run the complete suite once on the frozen final revision; record whether its
evidence is local or CI and whether it is still pending. Re-run after relevant
changes or failures, rather than repeating identical successful runs by default.

Keep README installation, execution, and test commands synchronized when tooling changes.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions/modules, and `PascalCase` classes. Ruff handles formatting and linting with a 100-character line length. Avoid unrelated formatting changes.

Use descriptive module and function names. For Markdown, use clear headings, fenced command examples, and relative links to repository files.

## Testing Guidelines

Tests use pytest with `test_*.py` discovery and are named by functional module. Add regression tests for behavior changes. Keep fixtures small, deterministic, and offline; mock public data access and explain external dependencies. Engineering fixture results do not establish predictive quality. Preserve completed experiment workspaces and their recorded source/runtime identities during upgrades.

## Commit & Pull Request Guidelines

Use concise imperative commit subjects. Submit changes on a temporary branch via a pull request targeting `main`; do not push code directly to `main`. Merge after required checks and review, then remove the temporary branch. Follow the contributor delivery guidance in [README.md](README.md).

Pull requests should describe the change, its purpose, and validation performed. Link related issues when available and explicitly identify checks that were not run. Keep public documentation, issues, and pull requests in English.

## Configuration & Output Discipline

Keep credentials, local environments, generated datasets, and large logs out of version control. Store verbose command output in files; share concise summaries and bounded excerpts when reporting results.
