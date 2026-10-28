# Version Control and Preserved Research History

The repository tracks source, tests, small fixtures, example profiles, dependency pins, documentation, and the reference paper. Local datasets, environments, runs, and session coordination files are not distributed by a clone.

## Branch and Pull Request Policy

Maintain one persistent branch, `main`. Implement changes on a temporary branch, submit a pull request, and merge after review and required checks pass. Delete the temporary local and remote branch names after merging, including the branch used for the current consolidation. Do not push code changes directly to `main`.

Before removing an older branch, verify that its tip is reachable from accepted `main` or an exact archival tag pushed to the remote. A renamed or selectively ported change is not proof that the original commit is retained. Archive tags preserve unadopted history without keeping extra active branches.

## Historical Candidates

The old multigeneration candidate at `689ecdf341ca` is preserved independently from the current engine. It contains older task/authority semantics and must not be merged wholesale into current forecast workflows. Its source fingerprint `7892a90385b252825511cf47b27d33e6bce296f69b652c6aaee7a49309ddc5a9` identifies a historical candidate, not the reorganized source tree.

The candidate's provenance records contain a 99-entry original manifest. Its Git snapshot excludes six generated egg-info files; of the remaining 93 paths, 92 match the frozen candidate byte for byte and `.gitignore` uses the repository rules. Access those records from the archived commit's `docs/provenance/` directory.

Historical offline regression and isolated generation-one replay records do not establish two consecutive semantic improvements, predictive quality, or production readiness. Those incomplete experiments remain paused. The small committed-pair recovery optimization from `56abd22116c5` is ported separately with its regression tests; that port does not accept the older experiments.

## Worktrees and Bound Runs

Preserve existing worktree directories, frozen commit IDs, private environments, datasets, and run artifacts. If deleting a branch checked out in a preserved worktree, detach that worktree at its unchanged commit first. Do not remove, clean, reset, or migrate its contents.

Source organization changes runtime identity. Resume or replay a historical run using its recorded source and dependency environment; never rewrite a run's identity to match new code. Local paths such as `runs/accepted_source/` may be absent from fresh clones and are not installation dependencies.

## Tracked and Ignored Files

Track `src/`, `tests/`, useful `examples/` and tools, `instruments/` profiles, `docs/`, dependency manifests, repository guides, and the reference PDF. Keep small JSON and model-text test fixtures tracked.

Ignore local environments, caches, build output, `data/`, `runs/`, logs, `.quant_rsi/` authority, coordination files, credentials, and machine configuration. Ignoring a path does not delete it. Git history does not replace backups of large datasets and execution evidence.

Before committing, inspect the diff and select only intended source, test, and documentation changes. Record validation for the actual candidate commit; historical fingerprints and earlier CI runs are not validation of a changed tree.
