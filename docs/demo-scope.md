# Demo Scope

## What The Public Demo Includes

- an anonymized/synthetic station dataset
- a saved artifact bundle for inference replay
- a FastAPI + Dash service
- public notebooks for EDA and replay
- helper scripts for validation, smoke testing, and notebook export

## What The Public Demo Does Not Claim

- real-time forecasting on live infrastructure data
- access to the internal source artifact used to produce the public demo artifact
- access to private ingestion logic or private API credentials
- a training-from-scratch workflow optimized for fast reruns on commodity hardware

## Historical Replay Constraint

Inference in this repo is intentionally constrained to the dataset window. Forecast start times must remain inside the replayable historical range so the demo can always overlay actual observations for evaluation.

That is a feature of the demo, not a limitation of the original work:

- it makes calibration visible
- it makes backtests easy to verify
- it avoids implying live deployment behavior that the public repo cannot reproduce

The same framing applies to calibration:

- the dashboard and replay notebook can fit a runtime calibration overlay using only the trailing `N` days before a chosen replay cutoff
- the comparison is causal with respect to that replay cutoff
- the result should be interpreted as a replay calibration simulation, not a live online system

## Active Station Scope

The public service is locked to a small set of demo stations:

- `Station_1`
- `Station_2`
- `Station_3`
- `Station_4`
- `Station_7`
- `Station_8`

`Station_6` is intentionally excluded from the public scope.

## Private Inputs That Should Stay Private

The following should not be part of the final public repository:

- `Final Internship Report.pdf`
- any rename maps connecting anonymized station names back to internal identifiers
- source artifacts generated from internal data
- operational credentials, access tokens, or ingestion notebooks tied to private systems

## Recommended Public Positioning

The cleanest public framing is:

"This repo demonstrates the design and packaging of a calibrated forecasting pipeline for heteroscedastic time series. The public service is a historical replay over an anonymized dataset so that the interval behavior can be inspected and reproduced."
