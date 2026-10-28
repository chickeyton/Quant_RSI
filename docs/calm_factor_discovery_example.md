# CALM egg-price factor tree: first worked example

Status: specification example, 2026-10-28. This is the first complete instance of
multi-branch factor discovery. It is not a runnable plan, not a fitted model,
and not evidence that CALM is predictable or tradable. The current engine still
accepts only a flat `(operator, window)` factor on one series. See issue
[#8](https://github.com/natureofnature/Quant_RSI/issues/8).

The machine-readable tree is
[`examples/calm_factor_tree.json`](../examples/calm_factor_tree.json). Nodes
there use the same ids as this document. Neither file is accepted by
`quant-rsi run`.

## What this example fixes

The target is the next move in Cal-Maine Foods (`CALM`). The primary chain is:

```text
CALM price
└─ wholesale egg price
   ├─ corn price
   ├─ soybean-meal price
   ├─ petroleum / diesel price
   ├─ milk price                  (registered hypothesis, relationship unresolved)
   └─ egg supply
      ├─ season
      ├─ temperature
      ├─ demand
      └─ H5N1 / HPAI
         └─ unresolved; the next generation proposes the predictor
```

Parentheses in the original brief are siblings. They are registered at the same
node and are not expanded just because the primary child is expanded.

## Procedure every later example repeats

1. Register one claim. Do not score it yet.
2. Split that claim into the drivers known at registration time, including any
   stated numeric value.
3. Bind a public series when one exists, and record when it becomes visible.
4. A driver with no series that is public before the decision stays open. Do
   not invent a formula to close it.
5. Ask what would predict that open driver, and bind evidence for those
   subfactors under the same rule.
6. Seal the tree, including every rejected and unresolved branch, before any
   score is computed.

Changing the subject, from eggs to rates or employment, changes the claims and
the series. It does not change these steps.

## Two gates before a series counts

**Availability.** A series used to predict time `t` must have been published
before the decision at `t`. A monthly layer count can explain an egg price
only after that month's report exists. A confirmed cull explains the supply
loss after confirmation, not before.

**Held-out confirmation.** The branch was chosen after seeing some history, so
that same history cannot confirm it. Every opened branch counts, including
branches that are dropped. Oil, corn, milk, season, temperature, and demand
can be attached together; at least one will match egg prices on the searched
sample. Only a branch that still helps on data not used to choose it is kept.

Opening more branches finds more published numbers. It does not by itself
prove the chain. Search stops when a qualified series is bound, when the
registered budget is exhausted, or when the node is explicitly left
unresolved. A convenient in-sample correlation is not a stopping condition.

## An unresolved node is a valid stop

`H5N1` is the open leaf of this example. Confirmed depopulation explains a
later drop in layer supply. Splitting it further does not guarantee a series
that signals the next outbreak before it happens. This generation therefore
stops that node at `unresolved_for_next_generation`.

The next generation inherits the open question, the bound siblings, and the
record of what was already tried. Its job is to propose how to predict that
node. It may bind a leading series whose publication time is before the
decision, such as wild-bird detections, nearby-premises confirmations, or
migratory season. It may also return the same unresolved status. It may not
relabel the post-confirmation cull count as a pre-outbreak signal, and it may
not fill the gap with an unregistered disease model.

Incomplete prediction is still the intended advantage. A forecast conditioned
on the bound series (egg price, lagged layer counts, confirmed culls, corn)
uses information that a forecast conditioned on nothing does not. That
advantage does not require every leaf to be solved, and it is not a claim of
a complete or profitable prediction. Whether the bound series actually improve
a pre-registered score is a later measurement, under the held-out gate above
and the still-open calibration work in issues
[#4](https://github.com/natureofnature/Quant_RSI/issues/4),
[#11](https://github.com/natureofnature/Quant_RSI/issues/11), and
[#12](https://github.com/natureofnature/Quant_RSI/issues/12).

## Node ledger

| Node | Role | Evidence | Cadence and visibility | This generation |
| --- | --- | --- | --- | --- |
| `calm_price` | target | Exchange daily bars | Daily. `CALM` is NASDAQ (`NMS`). Current forecast collection accepts NYQ, PCX, and ASE only, so this example cannot use `forecast run` as written. | Root claim |
| `egg_price` | primary driver of `calm_price` | [USDA AMS egg market reports](https://www.ams.usda.gov/market-news/egg-market-news-reports), including the daily national shell-egg index | Daily and weekly USDA reports. Urner Barry is the commercial index named in Cal-Maine's pricing description; it is not a free bound series here. | Bind the USDA series. Do not treat a commercial index as collected. |
| `corn_price` | sibling input to `egg_price` | Chicago Board of Trade corn daily settlement | Daily, after the session settlement is published | Bound leaf. Optional next-generation children are listed below and are not expanded now. |
| `soybean_meal_price` | sibling input to `egg_price` | Chicago Board of Trade soybean-meal settlement | Daily, after settlement | Bound leaf. Cal-Maine names soybean meal as the second feed ingredient. |
| `petroleum_price` | sibling hypothesis for `egg_price` | [EIA Weekly Petroleum Status Report](https://www.eia.gov/petroleum/supply/weekly/) and weekly retail diesel prices | Weekly, on the published EIA clock | Bound as a candidate cost series. The egg-price relationship is not established. |
| `milk_price` | sibling hypothesis for `egg_price` | USDA milk prices exist, but Cal-Maine's shell-egg discussion names egg quotes, flock size, demand, season, and HPAI, not milk | Usually not daily | `unresolved_relationship`. Do not expand it into a causal chain in this generation. |
| `egg_supply` | primary driver of `egg_price` | [NASS Chickens and Eggs](https://esmis.nal.usda.gov/publication/chickens-and-eggs): layers, eggs per 100 layers, production | Monthly. A release covers the preceding month and is issued around the third week of the next month. | Bind with the publication lag. The current month's unpublished count is not visible. |
| `season` | sibling of the supply question | Calendar | Known before the decision | Bound leaf |
| `temperature` | sibling of the supply question | NOAA daily station observations | Daily, after the observation day is published | Bound leaf candidate. The station set is not chosen here. |
| `demand` | sibling of the supply question | Holiday calendar is known in advance. Realized retail movement is usually late. | Mixed | Calendar is a leaf. Realized demand stays `unresolved_no_admissible_series`. |
| `h5n1` | primary driver of `egg_supply` | [APHIS commercial and backyard HPAI confirmations](https://www.aphis.usda.gov/livestock-poultry-disease/avian/avian-influenza/hpai-detections/commercial-backyard-flocks): confirmation date and birds affected | Irregular events. Visible only after confirmation. | `unresolved_for_next_generation` |

## Why these edges are registered

Cal-Maine's fiscal 2026 Form 10-K describes the company as pricing conventional
shell eggs from independently quoted wholesale markets, including Urner Barry
and USDA quotes, and states that those prices move with consumer demand,
season, the number and productivity of laying hens, and HPAI. Feed is
described as a primary production cost, with corn the main ingredient and
soybean meal the lesser one. The filing is a source for the edges, not a
measurement that those edges predict the next CALM return.

- Filing: [calm2026053010K](https://www.sec.gov/Archives/edgar/data/16160/000156276226000080/calm2026053010K.htm)
- CIK `0000016160`

Corn and soybean meal are feed costs. Petroleum is a transport and energy
hypothesis and stays marked as such. Milk is retained because the example
asked for it, with the relationship unresolved. Dairy-cattle H5N1 reports are
a possible child of the virus node, not evidence that the milk price moves
the egg price.

## What the next generation may do with `h5n1`

It proposes one pre-registered predictor and binds only series that are public
before the decision. Candidates already identified, none of which is selected
or scored by this example:

- [APHIS wild-bird HPAI detections](https://www.aphis.usda.gov/livestock-poultry-disease/avian/avian-influenza/hpai-detections/wild-birds).
  APHIS says the detection date is the confirmatory-test date, not the sample
  collection date. A generation that uses this series has to record that lag.
- Nearby commercial-premises confirmations from the same APHIS flock table.
  They explain a cull after confirmation; they are not automatically a leading
  indicator of the next premises.
- Migratory season, which is calendar information and can be bound now, but
  binding the calendar does not by itself predict an outbreak.
- [APHIS HPAI confirmations in livestock](https://www.aphis.usda.gov/livestock-poultry-disease/avian/avian-influenza/hpai-detections/hpai-confirmed-cases-livestock),
  as a spillover hypothesis rather than as the milk-price series.

If none of those is admissible before the decision, the node remains
unresolved and the rest of the tree can still be scored. The score must report
which nodes were unresolved so a reader can see the missing information.

## Optional deepening, not required to use a bound price

A daily settlement is already evidence for its own node. Predicting that
settlement is a separate question and is not required before the parent may
cite the settlement. The next generation may open these children; this
generation does not.

| Bound price | Possible later children | Public series | Cadence |
| --- | --- | --- | --- |
| `corn_price` | US corn supply and demand | [WASDE](https://www.usda.gov/about-usda/general-information/staff-offices/office-chief-economist/commodity-markets/wasde-report) | Monthly, on the published release clock |
| `corn_price` | crop condition and progress | [NASS Crop Progress](https://esmis.nal.usda.gov/publication/crop-progress) | Weekly in the growing season; monthly story in winter |
| `corn_price` | drought | US Drought Monitor | Weekly |
| `petroleum_price` | inventories and product supply | EIA Weekly Petroleum Status Report | Weekly |

Each of those children inherits the same two gates. A weather or progress
report that is published after the corn session being predicted does not
count.

## What this example does not do

- It does not authorize `forecast run CALM`. NASDAQ is outside the current
  exchange adapter.
- It does not change `FactorSpec`, the proposer, promotion, the cash
  benchmark, or the five-session objective.
- It does not treat a development gain on synthetic prices as evidence about
  CALM. See issue
  [#6](https://github.com/natureofnature/Quant_RSI/issues/6).
- It does not convert an egg-price error into a trading profit. Prediction
  quality and a costed strategy remain separate scores.
- It does not claim that enough branches will eventually produce a
  pre-outbreak H5N1 signal.
