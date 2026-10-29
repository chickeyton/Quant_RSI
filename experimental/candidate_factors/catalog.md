# Detailed candidate index

These 32 survey records are not runtime registrations. All have upstream execution
conformance `not_run` and predictive quality `not_assessed`. See the linked JSON for
parameters, units, state and integration requirements.

## Lean

[Full survey records](definitions/Lean.json)

| Candidate | Formula / behavior | Modes |
| --- | --- | --- |
| [`lean.rsi`](https://github.com/QuantConnect/Lean/blob/59a4205b1cb3c88a9c1bf37ffba9227c5ee9fdfa/Indicators/RelativeStrengthIndex.cs#L1-L126) | 100 - 100 / (1 + average_gain / average_loss). | recursive_state |
| [`lean.atr`](https://github.com/QuantConnect/Lean/blob/59a4205b1cb3c88a9c1bf37ffba9227c5ee9fdfa/Indicators/AverageTrueRange.cs#L1-L100) | Smoothed true range: max(high-low, abs(high-previous_close), abs(low-previous_close)). | recursive_state |
| [`lean.obv`](https://github.com/QuantConnect/Lean/blob/59a4205b1cb3c88a9c1bf37ffba9227c5ee9fdfa/Indicators/OnBalanceVolume.cs#L1-L99) | Add volume on higher close, subtract on lower close, hold on equal close. | recursive_state |
| [`lean.vwap`](https://github.com/QuantConnect/Lean/blob/59a4205b1cb3c88a9c1bf37ffba9227c5ee9fdfa/Indicators/VolumeWeightedAveragePriceIndicator.cs#L1-L117) | Rolling sum(volume * (open+high+low+close)/4) / sum(volume). | rolling_window |
| [`lean.ichimoku`](https://github.com/QuantConnect/Lean/blob/59a4205b1cb3c88a9c1bf37ffba9227c5ee9fdfa/Indicators/IchimokuKinkoHyo.cs#L1-L214) | Tenkan and Kijun are midpoints of rolling highs/lows; SenkouA averages delayed Tenkan/Kijun; SenkouB averages delayed rolling high/low; Chikou delays close. | rolling_window, shifted_components, multi_output |

## QuantDinger

[Full survey records](definitions/QuantDinger.json)

| Candidate | Formula / behavior | Modes |
| --- | --- | --- |
| [`quantdinger.earnings_yield`](https://github.com/OpenByteInc/QuantDinger/blob/8b2af21c288d87a5571d67ee6b2bfa860b06a4ba/backend_api_python/app/services/factors/registry.py#L407-L410) | Last net_income / last market_cap. | report_asof, elementwise |
| [`quantdinger.book_to_price`](https://github.com/OpenByteInc/QuantDinger/blob/8b2af21c288d87a5571d67ee6b2bfa860b06a4ba/backend_api_python/app/services/factors/registry.py#L407-L410) | Last shareholder_equity / last market_cap. | report_asof, elementwise |
| [`quantdinger.return_on_equity`](https://github.com/OpenByteInc/QuantDinger/blob/8b2af21c288d87a5571d67ee6b2bfa860b06a4ba/backend_api_python/app/services/factors/registry.py#L407-L410) | Last net_income / last shareholder_equity. | report_asof, elementwise |
| [`quantdinger.free_cash_flow_yield`](https://github.com/OpenByteInc/QuantDinger/blob/8b2af21c288d87a5571d67ee6b2bfa860b06a4ba/backend_api_python/app/services/factors/registry.py#L407-L410) | Last free_cash_flow / last market_cap. | report_asof, elementwise |
| [`quantdinger.revenue_growth`](https://github.com/OpenByteInc/QuantDinger/blob/8b2af21c288d87a5571d67ee6b2bfa860b06a4ba/backend_api_python/app/services/factors/registry.py#L421-L425) | Return last supplied revenue_growth value. | report_asof |
| [`quantdinger.amihud_illiquidity`](https://github.com/OpenByteInc/QuantDinger/blob/8b2af21c288d87a5571d67ee6b2bfa860b06a4ba/backend_api_python/app/services/factors/registry.py#L1078-L1084) | Mean(abs(simple_close_return) / (close * volume)). | rolling_window |
| [`quantdinger.parkinson_volatility`](https://github.com/OpenByteInc/QuantDinger/blob/8b2af21c288d87a5571d67ee6b2bfa860b06a4ba/backend_api_python/app/services/factors/registry.py#L1060-L1065) | sqrt(max(mean(log(high/low)^2)/(4*log(2)), 0) * 252). | rolling_window |
| [`quantdinger.supertrend`](https://github.com/OpenByteInc/QuantDinger/blob/8b2af21c288d87a5571d67ee6b2bfa860b06a4ba/backend_api_python/app/services/factors/registry.py#L717-L749) | ATR bands around (high+low)/2 with recursive trailing bands and direction switches. | recursive_state, multi_output |
| [`quantdinger.kama`](https://github.com/OpenByteInc/QuantDinger/blob/8b2af21c288d87a5571d67ee6b2bfa860b06a4ba/backend_api_python/app/services/factors/registry.py#L923-L939) | ER = abs(close[t]-close[t-period]) / sum(abs(close differences over period)); SC = (ER * (2/(fast+1)-2/(slow+1)) + 2/(slow+1))^2; KAMA[t] = KAMA[t-1] + SC * (close[t]-KAMA[t-1]). | recursive_state |
| [`quantdinger.vwap`](https://github.com/OpenByteInc/QuantDinger/blob/8b2af21c288d87a5571d67ee6b2bfa860b06a4ba/backend_api_python/app/services/factors/registry.py#L854-L866) | Rolling volume-weighted HLC3 price; output value or close/value - 1. | rolling_window |

## gs-quant

[Full survey records](definitions/gs-quant.json)

| Candidate | Formula / behavior | Modes |
| --- | --- | --- |
| [`gs_quant.sma`](https://github.com/goldmansachs/gs-quant/blob/e2b98f5fa0bfd3a931b9a6fcbee84c027588edea/gs_quant/timeseries/technicals.py#L56-L94) | Rolling arithmetic mean. | rolling_window |
| [`gs_quant.ema`](https://github.com/goldmansachs/gs-quant/blob/e2b98f5fa0bfd3a931b9a6fcbee84c027588edea/gs_quant/timeseries/technicals.py#L260-L295) | ema[t] = beta * ema[t-1] + (1-beta) * x[t], pandas ewm(adjust=False). | recursive_state |
| [`gs_quant.bollinger`](https://github.com/goldmansachs/gs-quant/blob/e2b98f5fa0bfd3a931b9a6fcbee84c027588edea/gs_quant/timeseries/technicals.py#L98-L142) | Moving average plus/minus k times rolling standard deviation. | rolling_window, multi_output |
| [`gs_quant.market_beta`](https://github.com/goldmansachs/gs-quant/blob/e2b98f5fa0bfd3a931b9a6fcbee84c027588edea/gs_quant/timeseries/econometrics.py#L966-L1055) | Rolling covariance(asset returns, benchmark returns) / rolling variance(benchmark returns). | rolling_window, multi_series_alignment |
| [`gs_quant.rolling_ols`](https://github.com/goldmansachs/gs-quant/blob/e2b98f5fa0bfd3a931b9a6fcbee84c027588edea/gs_quant/timeseries/statistics.py#L1173-L1256) | RollingOLS after dropping nonfinite rows and aligning X/y; exposes coefficients, fit statistics and fitted values. | rolling_fit, multi_series_alignment, multi_output |
| [`gs_quant.seasonal_trend`](https://github.com/goldmansachs/gs-quant/blob/e2b98f5fa0bfd3a931b9a6fcbee84c027588edea/gs_quant/timeseries/technicals.py#L539-L577) | **Blocked for point-in-time use.** Trend component of seasonal_decompose using its centered moving-average filter. | centered_noncausal |

## qlib

[Full survey records](definitions/qlib.json)

| Candidate | Formula / behavior | Modes |
| --- | --- | --- |
| [`qlib.kmid`](https://github.com/microsoft/qlib/blob/54355232463878d2eebb91fe0ee5fa7fa1f5976c/qlib/contrib/data/loader.py#L73-L310) | (close - open) / open | elementwise |
| [`qlib.klen`](https://github.com/microsoft/qlib/blob/54355232463878d2eebb91fe0ee5fa7fa1f5976c/qlib/contrib/data/loader.py#L73-L310) | (high - low) / open | elementwise |
| [`qlib.roc`](https://github.com/microsoft/qlib/blob/54355232463878d2eebb91fe0ee5fa7fa1f5976c/qlib/contrib/data/loader.py#L73-L310) | close[t - d] / close[t] | rolling_window |
| [`qlib.ma`](https://github.com/microsoft/qlib/blob/54355232463878d2eebb91fe0ee5fa7fa1f5976c/qlib/contrib/data/loader.py#L73-L310) | Mean(close, d) / close[t] | rolling_window |
| [`qlib.beta`](https://github.com/microsoft/qlib/blob/54355232463878d2eebb91fe0ee5fa7fa1f5976c/qlib/contrib/data/loader.py#L73-L310) | Slope(close, d) / close[t] | rolling_window |
| [`qlib.rsqr`](https://github.com/microsoft/qlib/blob/54355232463878d2eebb91fe0ee5fa7fa1f5976c/qlib/contrib/data/loader.py#L73-L310) | Rsquare(close, d) | rolling_window |
| [`qlib.resi`](https://github.com/microsoft/qlib/blob/54355232463878d2eebb91fe0ee5fa7fa1f5976c/qlib/contrib/data/loader.py#L73-L310) | Resi(close, d) / close[t] | rolling_window |
| [`qlib.rank`](https://github.com/microsoft/qlib/blob/54355232463878d2eebb91fe0ee5fa7fa1f5976c/qlib/contrib/data/loader.py#L73-L310) | Rank(close, d) | rolling_window |
| [`qlib.corr`](https://github.com/microsoft/qlib/blob/54355232463878d2eebb91fe0ee5fa7fa1f5976c/qlib/contrib/data/loader.py#L73-L310) | Corr(close, Log(volume + 1), d) | rolling_window |
| [`qlib.cord`](https://github.com/microsoft/qlib/blob/54355232463878d2eebb91fe0ee5fa7fa1f5976c/qlib/contrib/data/loader.py#L73-L310) | Corr(close / Ref(close, 1), Log(volume / Ref(volume, 1) + 1), d) | rolling_window |
| [`qlib.csrank`](https://github.com/microsoft/qlib/blob/54355232463878d2eebb91fe0ee5fa7fa1f5976c/qlib/data/dataset/processor.py#L326-L359) | (average_tie_percentile_rank_within_datetime - 0.5) * 3.46 | cross_section |
