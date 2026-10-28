# CALM egg-price factor tree: first worked example

Status: specification example, 2026-10-28. This is the first complete instance of
multi-branch factor discovery. It is not a runnable plan, not a fitted model,
and not evidence that CALM is predictable or tradable. The current engine still
accepts only a flat `(operator, window)` factor on one series. See issue
[#8](https://github.com/natureofnature/Quant_RSI/issues/8).

The machine-readable tree is
[`instruments/US/CALM/factor_tree.json`](../instruments/US/CALM/factor_tree.json). Nodes
there use the same ids as this document. Neither file is accepted by
`quant-rsi run`.

## What this example fixes

The target is the next move in Cal-Maine Foods (`CALM`). The primary chain is:

```text
CALM price
├─ wholesale egg price
│  ├─ egg supply
│  │  ├─ layer flock and lay rate          (NASS monthly, lagged)
│  │  ├─ pullets and chicks hatched        (future supply, NASS monthly)
│  │  ├─ weekly shell-egg inventory
│  │  ├─ cold storage
│  │  ├─ temperature
│  │  └─ H5N1 / HPAI
│  │     └─ unresolved; the next generation proposes the predictor
│  ├─ egg demand
│  │  ├─ season
│  │  │  ├─ Easter                         (date known in advance)
│  │  │  ├─ Thanksgiving
│  │  │  └─ Christmas
│  │  ├─ grocery-store feature activity    (advertised specials)
│  │  ├─ shell-egg demand indicator
│  │  ├─ breaking-stock demand
│  │  └─ foodservice demand                (no bound public series)
│  ├─ corn price
│  ├─ soybean-meal price
│  ├─ petroleum price                      (cost pass-through hypothesis)
│  └─ milk price                           (relationship unresolved)
├─ company freight                         (diesel; transport cost of the firm)
├─ labor
│  ├─ industry employment and wages        (BLS monthly, lagged)
│  ├─ headcount in filings                 (quarterly, lagged)
│  └─ job postings                         (no point-in-time archive)
├─ sales volume
│  ├─ dozens sold in filings               (quarterly, lagged)
│  └─ retail scanner movement              (paid source, not bound)
├─ product mix                             (conventional vs specialty, lagged filing)
└─ market move                             (control, not an egg cause)
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
sample. The same applies to freight, hiring, retail features, and holidays.
Only a branch that still helps on data not used to choose it is kept.

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
on the bound series (egg price, lagged layer counts, confirmed culls, corn,
holiday calendar, diesel, retail feature activity)
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
| `egg_price` | primary revenue driver | [USDA AMS egg market reports](https://www.ams.usda.gov/market-news/egg-market-news-reports) | Daily and weekly USDA reports. Urner Barry is the commercial index named by the issuer; it is not a free bound series. | Bind the USDA series |
| `egg_supply` | primary driver of `egg_price` | Supply is the parent of the rows below. It has no single series of its own. | Mixed | Split; do not score the parent without its children |
| `layer_flock` | current supply | [NASS Chickens and Eggs](https://esmis.nal.usda.gov/publication/chickens-and-eggs): layers, eggs per 100 layers, production | Monthly, for the preceding month, around the third week of the next month | Bind with the lag. The unpublished current month is not visible |
| `pullet_pipeline` | future supply | Same NASS report: pullets added, chicks hatched, eggs in incubators | Monthly, same lag | Bind as a leading supply series, still lagged |
| `weekly_shell_inventory` | near-term supply | USDA AMS Weekly Shell Egg Inventory | Weekly | Bind |
| `cold_storage` | inventory overhang | USDA AMS Weekly Cold Storage, selected centers | Weekly | Bind as a candidate. Coverage is selected centers, not the national flock |
| `temperature` | lay-rate input | NOAA daily station observations | Daily, after the observation day is published | Bound candidate. Station set is not chosen |
| `h5n1` | supply shock | [APHIS flock confirmations](https://www.aphis.usda.gov/livestock-poultry-disease/avian/avian-influenza/hpai-detections/commercial-backyard-flocks) | Irregular. Visible only after confirmation | `unresolved_for_next_generation` |
| `own_farm_hpai` | company-specific shock | Company 8-K or press release if it discloses a farm depopulation | After the company publishes it | Lagged event. Absence of a filing is not proof of no outbreak |
| `egg_demand` | primary driver of `egg_price`, sibling of supply | Parent of the demand rows. Demand is not the same object as supply. | Mixed | Split |
| `season` | demand calendar | Civil calendar | Known before the decision | Bound parent of the holidays |
| `easter` | holiday demand | Western Easter Sunday, computable for any future year | Known in advance | Bound leaf. This is the main US shell-egg holiday |
| `thanksgiving` | holiday demand | Fourth Thursday in November | Known in advance | Bound leaf |
| `christmas` | holiday demand | 25 December | Known in advance | Bound leaf |
| `retail_feature_activity` | advertised supermarket specials | USDA AMS Weekly Grocery Store Egg Feature Activity | Weekly | Bind. Also informs `sales_volume` |
| `shell_demand_indicator` | wholesale demand tone | USDA AMS Weekly Shell Egg Demand Indicator | Weekly | Bind as a candidate tone series, not as realized retail volume |
| `breaking_stock` | alternative outlet for eggs | USDA AMS Daily National Breaking Stock Report | Daily | Bind |
| `foodservice_demand` | channel mix | No free decision-time series is bound | — | `unresolved_no_admissible_series` |
| `corn_price` | feed cost | Chicago Board of Trade corn settlement | Daily, after settlement | Bound leaf |
| `soybean_meal_price` | feed cost | Chicago Board of Trade soybean-meal settlement | Daily, after settlement | Bound leaf |
| `petroleum_price` | hypothesis that fuel passes into the egg quote | [EIA weekly petroleum and retail diesel](https://www.eia.gov/petroleum/gasdiesel/) | Weekly | Bound candidate. Relationship not established |
| `milk_price` | proposed sibling | Not named in the issuer's shell-egg discussion | — | `unresolved_relationship` |
| `company_freight` | direct cost of moving feed and eggs | Same EIA weekly retail diesel series | Weekly | Separate from `petroleum_price`: this edge is the firm's transport cost |
| `labor` | operating cost and capacity | Parent of the labor rows | Mixed | Split |
| `industry_wages` | labor market | BLS Current Employment Statistics for poultry processing: employment and hourly earnings | Monthly, published with the employment-situation lag | Bind with the lag. This is the industry, not Cal-Maine alone |
| `filing_headcount` | company labor | Employees disclosed in Form 10-K and 10-Q | Quarterly or annual, after filing | Bind with the lag |
| `job_postings` | hiring visible on the careers page | A live careers page is not a historical archive | Current page only | `unresolved_no_point_in_time_archive` |
| `sales_volume` | dozens moved, distinct from price | Parent of the sales rows | Mixed | Split |
| `filing_dozens_sold` | company volume | Dozens sold in 10-Q and 10-K | Quarterly, after filing | Bind with the lag |
| `retail_scanner` | supermarket movement and promoted price | Nielsen / IRI class scanner panels | Commercial | `unresolved_paid_source` |
| `product_mix` | conventional vs specialty | Mix disclosed in filings. Specialty is described as more often cost-based | Quarterly or annual | Bind with the lag |
| `market_move` | control | A broad US equity index such as the S&P 500 | Daily | Not an egg cause. A CALM move that only matches the index is not evidence for this tree |

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

Corn and soybean meal are feed costs. Diesel is registered twice, on purpose.
`petroleum_price` asks whether fuel changes the wholesale egg quote.
`company_freight` asks whether the same diesel series raises what Cal-Maine
pays to move feed and eggs. Those are different claims and both have to pass
the held-out gate. Milk stays unresolved. Dairy-cattle H5N1 reports are a
possible child of the virus node, not evidence that the milk price moves the
egg price.

Supply and demand are siblings under the egg price. Season is a child of
demand, and Easter, Thanksgiving, and Christmas are children of season because
their dates are known before the decision. A holiday can raise demand without
any new supply information. Retail feature activity is the public record of
advertised supermarket specials. It informs both demand and sales volume.
Paid scanner panels and a historical archive of job postings are named and
left unresolved. The BLS egg CPI is not a node: it restates the egg price
after the month is over, so it is not a driver of that price.

`market_move` is a control. CALM can rise because the equity market rises.
That observation stays outside the egg chain and cannot be counted as
evidence that an egg, freight, or labor node worked.

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
