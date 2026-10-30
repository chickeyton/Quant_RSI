# Third-party attribution for the candidate survey

This collection includes selected/reformatted metadata, formula templates and
source-derived descriptions. Changes made here: static extraction into JSON,
selection of candidate families, paraphrased explanations, execution-mode tags,
and original synthetic examples. No upstream executable module is bundled.
The referenced upstream content remains subject to its original license; this
file does not relicense Quant RSI or grant rights to uncollected dependencies,
market data, trademarks or third-party assets.

Exact repository URLs, commits, selected paths, source hashes and license artifact
hashes are in [sources.json](sources.json). Candidate records carry pinned source
links and line anchors. Keep this notice and the corresponding license files with
copies or adaptations of the collected material.

| Source | Attribution from selected sources | Preserved terms and notices |
| --- | --- | --- |
| Microsoft Qlib | Copyright (c) Microsoft Corporation. | [MIT LICENSE](licenses/qlib/LICENSE) |
| GS Quant | Copyright 2018 Goldman Sachs; see both upstream notices for additional attribution | [Apache-2.0 LICENSE](licenses/gs-quant/LICENSE), [NOTICE](licenses/gs-quant/NOTICE), [NOTICE.txt](licenses/gs-quant/NOTICE.txt) |
| QuantConnect LEAN | Copyright 2014 QuantConnect Corporation. | [Apache-2.0 LICENSE](licenses/Lean/LICENSE) |
| QuantDinger selected backend factor files | 2025-2026 Open Byte Inc. and QuantDinger contributors | [LICENSE](licenses/QuantDinger/LICENSE), [REUSE.toml](licenses/QuantDinger/REUSE.toml), [LICENSES/Apache-2.0.txt](licenses/QuantDinger/Apache-2.0.txt) |

The local Apache-2.0.txt preserves QuantDinger's upstream
`LICENSES/Apache-2.0.txt` bytes. REUSE.toml supplies the selected backend files'
copyright/license annotation. GS Quant's NOTICE and NOTICE.txt are both retained
verbatim, including the Scott Weinstein attribution. License conclusions are
limited to the selected paths at the pinned revisions, not all files in each
repository. The LEAN broad inventory contains path names only; executable reuse
of additional indicators requires a fresh per-file and dependency check.

Upstream adapter files referencing Qlib or TA-Lib are evidence about integration
patterns, not permission to redistribute those dependencies. The Apache-2.0
notice/attribution and MIT notice requirements continue to apply when material
moves out of this experimental directory.
