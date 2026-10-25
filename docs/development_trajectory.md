# Finite development trajectory

The trajectory connects a genuine local C2 research run to the existing C4
package experiment. It is a development controller with at most two proposal
stages and one semantic package change. It does not open protected C5 evidence,
advance an accepted release, place orders, or establish profitable self-improvement.

## Register the source and budget

Use the same installed runtime that produced the source C2 run. The source must
be standalone, use local event data and the `window_first` policy, and have no
pending research requests or unresolved attempts. Obtain its exact
`checkpoint_ref` from `quant-rsi research replay /absolute/path/to/source`.

Copy [the plan template](../examples/development_trajectory.json) and replace
`source_directory` and `source_checkpoint_ref`. Set an absolute
`QUANT_RSI_WORKSPACE`; this selects one stable trajectory authority, independent
of the current directory or plan display name.

```bash
export QUANT_RSI_WORKSPACE=/absolute/path/to/workspace
quant-rsi trajectory init /absolute/path/to/trajectory-plan.json
quant-rsi trajectory run --steps 1
quant-rsi trajectory resume --steps 12
quant-rsi trajectory replay
```

Initialization freezes the actual source files, data, checkpoint, model,
training masks, availability evidence and historical costs. An identical
initialization returns the existing authority; changed substantive inputs are
rejected. Subsequent work uses the frozen copy even if original source paths
or the market registry disappear.

## Understand finite progress and costs

Each outer step reserves or completes one durable operation. A C4 operation
has fixed parent/candidate targets. Recovery completes that original operation;
a committed child response is reconciled without another worker dispatch.
Legitimate failed trials remain in the research history and budget while the
original search may continue within its existing grant.

`worker_attempts` counts charged Kernel attempts. `registered_worker_seconds`
counts reserved timeout slots, not elapsed runtime. Initial source costs count
once. Stage capacity maps to the C4 grants and is not charged twice. Reports
separate confirmed consumption, unused irrevocable commitment and unresolved
child costs. Capacity is never refunded.

A completed, valid pair can advance only the experimental head from
`window_first` to `operator_first`. Negative returns or zero score differences
remain visible and do not become profitability claims. The second proposal
authenticates that actual candidate and records `no_patch: grammar_exhausted`
without workers, even when all capacity is occupied. First-stage `no_patch`,
insufficient capacity, or a terminal required-operation failure also seals the
trajectory.

## Inspect and recover

`status` and `replay` verify committed snapshots, including SQLite journals
replayed in memory, selected models and training masks. They do not fit, propose,
repair artifacts, or consult mutable original inputs. Missing or corrupt
evidence fails validation. A pending child may already have consumed capacity;
the report keeps its unused commitment reserved until reconciliation.

Use `trajectory resume` for delegated work. Calling a delegated C4/C2 entry
directly requires the actual owning parent locks and matching durable targets;
copied context metadata or omitted context does not confer execution authority.
Keep the entire `.quant_rsi/trajectory` directory and sibling authority anchor
when preserving a run. Deleting or editing them is not a quota reset procedure.
