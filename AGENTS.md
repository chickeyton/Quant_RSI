# Repository Guidelines

## Project Structure & Module Organization

This workspace currently contains one reference document, `2505.15155v2.pdf`. There are no source modules, tests, assets directories, or dependency manifests yet. Preserve the reference PDF unless replacing it is an explicit part of the contribution.

As implementation is introduced, place source code in `src/`, automated tests in `tests/`, and runnable examples in `examples/`. Add a `README.md` explaining the project purpose, prerequisites, and directory layout. Create directories only when they contain useful files.

## Build, Test, and Development Commands

No build, test, or local execution commands are configured. Do not assume commands such as `npm test`, `pytest`, or `make build` are supported.

The first implementation should document exact dependency installation, execution, and test commands in `README.md`, with required runtime versions. Keep this guide synchronized with those commands when tooling is added.

## Coding Style & Naming Conventions

No programming language, formatter, or linter has been selected. Use the chosen language’s standard conventions and configure its formatter before establishing a large codebase. Keep indentation consistent within each file and avoid unrelated formatting changes.

Use descriptive module and function names. For Markdown, use clear headings, fenced command examples, and relative links to repository files.

## Testing Guidelines

No testing framework or coverage threshold exists. Introduce automated tests alongside executable code and document how to run them. Mirror source organization under `tests/` and follow the selected framework’s discovery conventions. Keep test inputs small and deterministic; explain any external dependencies.

## Commit & Pull Request Guidelines

This directory is not currently a Git repository, so no historical commit convention can be inferred. Once version control is initialized, use concise, imperative commit subjects, such as `Add reference implementation`.

Pull requests should describe the change, its purpose, and validation performed. Link related issues when available and explicitly identify checks that were not run.

## Configuration & Output Discipline

Keep credentials, local environments, generated datasets, and large logs out of version control. Store verbose command output in files; share concise summaries and bounded excerpts when reporting results.
