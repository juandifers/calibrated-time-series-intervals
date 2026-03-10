# Design Decisions

## Goal

The core objective was to build a forecasting pipeline whose uncertainty estimates remain useful when stations have very different flow scales, periodicity, and noise profiles. In this setting, point forecasts alone are not enough. The real design target is calibrated, explainable interval behavior.

## Why Direct Multi-Output Forecasting

The model predicts the full 96-step horizon directly instead of recursively feeding predictions back into the next step.

Why:

- recursive forecasting compounds error over a full day horizon
- each lead time has a different difficulty profile
- direct training makes calibration easier because every horizon step has a stable residual distribution

## Why Station-Aware Scaling

Stations differ substantially in level and volatility. A single global scaling rule would make residuals harder to compare and would distort interval widths.

Using a separate robust scaler per station helps:

- normalize the regression target without removing station identity
- keep low-flow and high-flow stations numerically comparable
- make conformal residual bins more stable

## Why Mondrian Keys Use Station + Horizon + Time Of Day

The public demo uses Mondrian keys of the form:

`(station, horizon_step, local_time_of_day_bin)`

Each component carries different uncertainty information:

- `station`: some stations are intrinsically noisier than others
- `horizon_step`: short-range and long-range forecast errors behave differently
- `time_of_day_bin`: demand variance often changes with intraday usage patterns

This lets the interval width adapt to heteroscedasticity instead of averaging it away.

## Why Hierarchical Backoff

The most specific bins are not always well populated. If a `(station, horizon, time_of_day)` bucket is sparse, the demo falls back to broader contexts.

Fallback order:

1. `(station, horizon, time_of_day)`
2. `(station, horizon)`
3. `(station,)`
4. pooled/global bins

This gives a practical compromise:

- dense contexts stay specific
- sparse contexts stay stable
- the service can still serve intervals without brittle failures

## Why Historical Replay Instead Of Live Inference

For a public portfolio repo, replay has better tradeoffs than pretending to be live:

- reviewers can verify actuals against the forecast interval
- the service behavior is deterministic
- there is no need to expose private APIs or operational credentials
- the notebook and dashboard can demonstrate both forecast generation and backtesting with the same artifact bundle

## Why Keep Runtime Calibration Separate

The demo supports manual calibration overlays written to `runtime/calibration_states/`. Those updates are intentionally kept outside the immutable base artifact.

That separation makes it easier to explain:

- what came from offline training
- what changed during runtime calibration
- how interval behavior can adapt without mutating the original artifact snapshot

- the repo is about reliability and uncertainty, not just forecast curves
- the design choices are motivated by data behavior, not by model novelty alone
- the system includes packaging, validation, replay, and service interfaces, not only notebooks
