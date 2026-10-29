# Candidate price adapters v1

Two close-only candidates from the [source-pinned survey](../experimental/candidate_factors/README.md)
now execute as shared numerical leaves in [bounded factor discovery](factor_discovery.md).
They use the existing point-in-time inputs, finite expression grammar, fit allocation,
Journal and forecast worker. They do not create a second registry or evaluation path.

## Frozen semantics

| Typed recipe `op` | Survey candidate | Output at session t | Required observations |
| --- | --- | --- | --- |
| `qlib_roc_v1` | `qlib.roc` | `close[t-d] / close[t]` | d + 1 consecutive closes |
| `qlib_ma_v1` | `qlib.ma` | `mean(close[t-d+1:t+1]) / close[t]` | d consecutive closes |

Both outputs are dimensionless **levels**, not percentage returns. In particular,
Qlib ROC is an inverse price ratio: closes 10, 20, 40 with d=2 produce 0.25,
not conventional momentum 3 or 300%. A constant path gives 1 for both adapters.
The `window` d is an integer from 1 through 30. MA includes the current close;
MA with d=1 is exactly 1. It is not a price-unit moving average.

The recipes retain the exact upstream revision
`54355232463878d2eebb91fe0ee5fa7fa1f5976c`. The formula source is
[Qlib Alpha158DL.get_feature_config](https://github.com/microsoft/qlib/blob/54355232463878d2eebb91fe0ee5fa7fa1f5976c/qlib/contrib/data/loader.py#L73-L310).
The survey retains its [third-party notices](../experimental/candidate_factors/THIRD_PARTY_NOTICES.md)
and [MIT license](../experimental/candidate_factors/licenses/qlib/LICENSE).
These are local implementations of those two formulas; no Qlib code is executed.

Unlike an upstream operator that might emit partial-window results, these adapters
require every finite, strictly positive close in the registered full window. A
missing, late, nonpositive or unrepresentable value yields missing, never zero or
an imputed price. The existing input path also enforces membership, quote quality,
calendar-close alignment, availability and corporate-action qualifications. No
source is collected implicitly. Future observations cannot enter an earlier origin.
The existing common registered-leaf eligibility policy remains in effect.

These rolling adapters have no mutable recursive state: each result is computed
from its frozen as-of history. Window parameters, operation version and source
revision are preserved in each leaf, catalog binding, discovery plan and worker
recipe identity. Changing a parameter creates a different recipe. Legacy
`FactorSpec` and its serialized momentum/mean-return recipes remain unchanged.
As with every source upgrade, preserve completed workspaces and their original
source/dependency environments; do not rewrite their runtime fingerprints.

## Run through shared definitions and explicit bindings

The repository-owned [candidate_prices catalog](../instruments/shared/candidate_prices.json)
contains reusable five-window definitions. The [KO profile](../instruments/US/KO/factors.json)
explicitly binds them to its close-price input. Another instrument can select the
same shared definitions with its own bindings; there is no implicit universe-wide
activation. Catalog resolution alone still reports `specification_only`.

From the repository's Linux/Python 3.12 editable environment, use an existing
`ForecastInputs` envelope for KO:

```bash
python examples/prepare_factor_discovery.py /absolute/path/to/inputs.json \
  --output runs/ko-candidate-inputs --campaign-id ko-candidate-prices \
  --factor-set candidate-prices --exposure historical_market_exploration

export QUANT_RSI_WORKSPACE="$(pwd)/runs/ko-candidate-workspace"
quant-rsi factor-discovery init runs/ko-candidate-inputs/plan.json \
  --inputs runs/ko-candidate-inputs/inputs.json
quant-rsi factor-discovery resume ko-candidate-prices --max-fits 1
quant-rsi factor-discovery status ko-candidate-prices
quant-rsi factor-discovery replay ko-candidate-prices
```

The example defaults to the legacy factors unless `--factor-set candidate-prices`
is supplied. Its default grid reserves three candidate rounds, three folds and
21 fits total; each resume bounds newly dispatched fits. The small offline test
grid instead uses `--folds '[[80],[90]]' --exposure engineering_fixture` with an
appropriate test envelope. Repeating a completed campaign replays its sealed
artifacts without new fits. Configuration JSON lives in the checkout, as do the
existing instrument profiles; installed Python modules do not search for it.

## Evidence and remaining work

Tests cover independent arithmetic, exact warmup boundaries, source pinning,
legacy serialization, invalid/missing/late inputs, future-data mutations and
actual B0/D1 LightGBM execution for each adapter, including interrupted/resumed
fit budgets and sealed replay. They are deterministic offline engineering tests.
Full upstream-runtime conformance remains **not run**; predictive usefulness is
**not assessed**. Neither passing fixtures nor admission into discovery establishes
alpha, profitability or independent confirmation.

The other 30 survey candidates remain specifications. OHLCV, cross-sectional,
report-as-of, multi-output and recursive-state adapters require their own input
and execution contracts; noncausal candidates remain excluded. EMA/RSI would need
explicit seed, state, reset and chunked-replay behavior. Research-policy evolution,
prospective outcomes and statistical confirmation are separate follow-ups.

Implements [#35](https://github.com/natureofnature/Quant_RSI/issues/35), following
[#31](https://github.com/natureofnature/Quant_RSI/issues/31) and survey
[#33](https://github.com/natureofnature/Quant_RSI/pull/33).
