#!/usr/bin/env python3
"""Validate anonymized demo artifact invariants."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate the demo artifact")
    parser.add_argument("--artifact", type=Path, default=repo_root / "mondrian_artifacts_demo")
    parser.add_argument(
        "--source-artifact",
        type=Path,
        default=repo_root / "mondrian_artifacts",
        help="Optional private source artifact used for extra consistency checks.",
    )
    return parser.parse_args()


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _assert(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def main() -> None:
    args = parse_args()
    artifact = args.artifact.resolve()
    source_artifact = args.source_artifact.resolve()
    if not artifact.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact}")

    model_path = artifact / "model" / "xgb_direct_multioutput.joblib"
    _assert(
        model_path.exists(),
        "Demo model artifact is missing. Run `python scripts/fetch_demo_model.py` first.",
    )

    station_cfg = _load_json(artifact / "meta" / "station_config.json")
    active = station_cfg.get("active_stations", [])
    excluded = station_cfg.get("excluded_stations", [])

    _assert(sorted(active) == ["Station_1", "Station_2", "Station_3", "Station_4", "Station_7", "Station_8"], "Active stations do not match locked allowlist")
    _assert(excluded == ["Station_6"], "Excluded stations must contain only Station_6")
    _assert(int(station_cfg.get("horizon", -1)) == 96, "station_config horizon must be 96")

    meta = _load_json(artifact / "meta" / "meta.json")
    _assert(int(meta.get("horizon", -1)) == 96, "meta horizon must be 96")

    text_files = [
        artifact / "calibrator" / "quantiles.json",
        artifact / "calibrator" / "counts.json",
        artifact / "calibrator" / "station_profiles.json",
        artifact / "transforms" / "station_ohe_columns.json",
    ]
    for path in text_files:
        txt = path.read_text(encoding="utf-8")
        _assert(re.search(r"GES\\d", txt) is None, f"Found leaked old station ID in {path}")

    quantiles = _load_json(artifact / "calibrator" / "quantiles.json")
    qL = quantiles.get("qL", {})
    qU = quantiles.get("qU", {})
    _assert(set(qL) == set(qU), "qL and qU key sets must match")

    seen_h = set()
    for key in qL:
        obj = ast.literal_eval(key)
        if isinstance(obj, tuple):
            tags = dict(obj)
            station = tags.get("S")
            h = tags.get("H")
            if station is not None:
                _assert(station in active, f"Unexpected station in quantiles key: {station}")
            if h is not None:
                seen_h.add(int(h))
    _assert(seen_h == set(range(96)), "Station quantile horizon keys must cover H=0..95")

    station_ohe_index = _load_json(artifact / "transforms" / "station_ohe_index.json")
    _assert(sorted(station_ohe_index.keys()) == sorted(active), "station_ohe_index keys must match active stations")

    demo_ohe = _load_json(artifact / "transforms" / "station_ohe_columns.json")
    ohe_dim = int(_load_json(artifact / "transforms" / "ohe_dim.json").get("ohe_dim", -1))
    _assert(len(demo_ohe) == ohe_dim, "Demo OHE columns length mismatch with ohe_dim")
    if (source_artifact / "transforms" / "station_ohe_columns.json").exists():
        source_ohe = _load_json(source_artifact / "transforms" / "station_ohe_columns.json")
        _assert(len(source_ohe) == ohe_dim, "Demo ohe_dim must equal source artifact OHE dim")
        print("[OK] Source artifact consistency checks passed")
    else:
        print(f"[OK] Skipping source artifact checks (not found: {source_artifact})")

    print("[OK] Demo artifact validation passed")
    print(f"[OK] Active stations: {', '.join(active)}")
    print(f"[OK] OHE dim: {ohe_dim}")


if __name__ == "__main__":
    main()
