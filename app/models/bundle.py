from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)


class DirectMultiOutputXGB:
    """Compatibility class for unpickling training artifacts."""

    def __init__(self, models: list[Any]):
        self.models = models
        self.H = len(models)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        preds = []
        for model in self.models:
            booster = model.get_booster()
            yhat = booster.inplace_predict(X, predict_type="value")
            preds.append(yhat)
        return np.column_stack(preds).astype(np.float32)


sys.modules["__main__"].DirectMultiOutputXGB = DirectMultiOutputXGB


@dataclass
class DemoBundle:
    model: Any
    meta: dict[str, Any]
    qL: dict[str, float]
    qU: dict[str, float]
    bins: dict[str, dict[str, float]]
    counts: dict[str, Any]
    station_scalers: dict[str, Any]
    station_ohe_index: dict[str, int]
    ohe_dim: int
    station_ohe_columns: list[str]


def load_demo_bundle(artifact_dir: Path) -> DemoBundle:
    artifact_dir = Path(artifact_dir)

    model_path = artifact_dir / "model" / "xgb_direct_multioutput.joblib"
    meta_path = artifact_dir / "meta" / "meta.json"
    quantiles_path = artifact_dir / "calibrator" / "quantiles.json"
    counts_path = artifact_dir / "calibrator" / "counts.json"
    scalers_path = artifact_dir / "transforms" / "station_scalers.joblib"
    ohe_index_path = artifact_dir / "transforms" / "station_ohe_index.json"
    ohe_dim_path = artifact_dir / "transforms" / "ohe_dim.json"
    ohe_cols_path = artifact_dir / "transforms" / "station_ohe_columns.json"

    required = [
        model_path,
        meta_path,
        quantiles_path,
        counts_path,
        scalers_path,
        ohe_index_path,
        ohe_dim_path,
        ohe_cols_path,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        if not model_path.exists():
            raise FileNotFoundError(
                "Demo model artifact is missing. "
                "Download it with `python scripts/fetch_demo_model.py` before starting the service. "
                f"Expected path: {model_path}"
            )
        raise FileNotFoundError(f"Missing artifact files: {missing}")

    try:
        model = joblib.load(model_path)
    except ModuleNotFoundError as exc:
        if exc.name == "xgboost":
            raise ModuleNotFoundError(
                "xgboost is required to load the demo model artifact. Install it with `pip install xgboost`."
            ) from exc
        raise
    if isinstance(model, list):
        model = DirectMultiOutputXGB(model)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    quantiles = json.loads(quantiles_path.read_text(encoding="utf-8"))
    counts = json.loads(counts_path.read_text(encoding="utf-8"))
    scalers = joblib.load(scalers_path)
    station_ohe_index = json.loads(ohe_index_path.read_text(encoding="utf-8"))
    ohe_dim_obj = json.loads(ohe_dim_path.read_text(encoding="utf-8"))
    ohe_cols = json.loads(ohe_cols_path.read_text(encoding="utf-8"))

    qL = {str(k): float(v) for k, v in quantiles.get("qL", {}).items()}
    qU = {str(k): float(v) for k, v in quantiles.get("qU", {}).items()}
    bins = {
        str(k): {
            "qL": float(v.get("qL", 0.0)),
            "qU": float(v.get("qU", 0.0)),
        }
        for k, v in quantiles.get("bins", {}).items()
        if isinstance(v, dict)
    }

    ohe_dim = int(ohe_dim_obj.get("ohe_dim", len(ohe_cols)))

    bundle = DemoBundle(
        model=model,
        meta=meta,
        qL=qL,
        qU=qU,
        bins=bins,
        counts=counts,
        station_scalers=scalers,
        station_ohe_index={str(k): int(v) for k, v in station_ohe_index.items()},
        ohe_dim=ohe_dim,
        station_ohe_columns=[str(c) for c in ohe_cols],
    )

    logger.info(
        "Loaded demo bundle from %s (horizon=%s, stations=%s, q_keys=%s, ohe_dim=%s)",
        artifact_dir,
        bundle.meta.get("horizon"),
        len(bundle.station_scalers),
        len(bundle.qL),
        bundle.ohe_dim,
    )
    return bundle
