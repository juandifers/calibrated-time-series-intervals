# Artifact Schema

The public demo service loads the artifact bundle under `mondrian_artifacts_demo/`.

## Directory Layout

### `model/`

- `xgb_direct_multioutput.joblib`

Saved forecasting model used for the fixed 96-step horizon.

### `transforms/`

- `station_scalers.joblib`
- `station_ohe_columns.json`
- `station_ohe_index.json`
- `ohe_dim.json`

Station-aware preprocessing metadata used to reconstruct the feature representation expected by the model.

### `calibrator/`

- `quantiles.json`
- `counts.json`
- `station_profiles.json`

Data needed to resolve interval widths. The quantiles support specific and fallback contexts. Counts and station profiles help validate scope and calibration coverage.

### `meta/`

- `meta.json`
- `station_config.json`
- `demo_manifest.json`

Metadata describing horizon, active station scope, and artifact provenance. In the public repo, provenance may reference private build inputs that are intentionally not included.

## Runtime State

Runtime updates are stored separately under `runtime/`.

### `runtime/calibration_states/`

Per-station calibration overlays written by the manual calibration flow. These files are mutable and should not be treated as part of the base artifact.

### `runtime/calibration_jobs.json`

Job status tracking for calibration runs.

## Why This Split Matters

Separating immutable artifact files from mutable runtime state makes the demo easier to reason about:

- base model and base conformal quantiles stay reproducible
- runtime calibration remains inspectable
- public replay behavior remains deterministic until a calibration job explicitly changes runtime state
