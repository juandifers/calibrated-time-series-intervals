#!/usr/bin/env python3
"""Build an anonymized demo artifact from a source Mondrian artifact."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sanitized demo artifact")
    parser.add_argument("--source-artifact", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--rename-map", required=True, type=Path)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_rename_map(markdown_path: Path) -> Dict[str, str]:
    text = markdown_path.read_text(encoding="utf-8")
    if "=" not in text:
        raise ValueError(f"Invalid rename map file: {markdown_path}")
    payload = text.split("=", 1)[1].strip()
    mapping = ast.literal_eval(payload)
    if not isinstance(mapping, dict):
        raise ValueError("Parsed rename map is not a dict")
    return {str(k): str(v) for k, v in mapping.items()}


def ensure_dirs(root: Path) -> None:
    for rel in [
        "model",
        "meta",
        "calibrator",
        "transforms",
        "states",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def parse_station_key(key: str) -> Tuple[bool, Any]:
    """Return (is_station_key, parsed_object_or_original)."""
    try:
        obj = ast.literal_eval(key)
    except Exception:
        return False, key

    if not isinstance(obj, tuple) or not obj or not isinstance(obj[0], tuple):
        return False, obj

    tags = {a: b for a, b in obj if isinstance(a, str)}
    return ("S" in tags), obj


def rewrite_station_key(obj: tuple, old_to_new: Dict[str, str]) -> str | None:
    tags = []
    found_station = False
    station_allowed = False

    for item in obj:
        if not (isinstance(item, tuple) and len(item) == 2):
            tags.append(item)
            continue
        k, v = item
        if k == "S":
            found_station = True
            if v not in old_to_new:
                return None
            tags.append(("S", old_to_new[v]))
            station_allowed = True
        else:
            tags.append((k, v))

    if found_station and not station_allowed:
        return None
    return str(tuple(tags))


def stable_json_dump(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_directory(path: Path) -> str:
    h = hashlib.sha256()
    files = sorted([p for p in path.rglob("*") if p.is_file()])
    for p in files:
        h.update(str(p.relative_to(path)).encode("utf-8"))
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def select_demo_stations(csv_path: Path, rename_map: Dict[str, str], excluded_new: set[str]) -> Tuple[list[str], list[str]]:
    df = pd.read_csv(csv_path, nrows=1)
    csv_stations = [c for c in df.columns if c != "timestamp" and not c.startswith("Unnamed")]
    active_new = sorted([s for s in csv_stations if s in set(rename_map.values()) and s not in excluded_new])
    new_to_old = {v: k for k, v in rename_map.items()}
    active_old = sorted([new_to_old[s] for s in active_new])
    return active_new, active_old


def main() -> None:
    args = parse_args()

    source = args.source_artifact.resolve()
    output = args.output.resolve()
    rename_map_path = args.rename_map.resolve()
    csv_path = args.csv.resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source artifact not found: {source}")

    rename_map = load_rename_map(rename_map_path)
    excluded_new = set(args.exclude)
    active_new, active_old = select_demo_stations(csv_path, rename_map, excluded_new)

    old_to_new_active = {old: new for old, new in rename_map.items() if old in set(active_old)}
    if not old_to_new_active:
        raise ValueError("No active stations selected for demo artifact")

    if output.exists():
        shutil.rmtree(output)
    ensure_dirs(output)

    # 1) model file copy
    shutil.copy2(source / "model" / "xgb_direct_multioutput.joblib", output / "model" / "xgb_direct_multioutput.joblib")

    # 2) meta patch
    meta = json.loads((source / "meta" / "meta.json").read_text(encoding="utf-8"))
    meta["horizon"] = 96
    meta["demo_artifact"] = True
    meta["active_stations"] = active_new
    meta["excluded_stations"] = sorted(excluded_new)
    meta["created_utc"] = datetime.now(timezone.utc).isoformat()
    stable_json_dump(meta, output / "meta" / "meta.json")

    # Export machine-readable station config
    station_config = {
        "active_stations": active_new,
        "excluded_stations": sorted(excluded_new),
        "horizon": 96,
    }
    stable_json_dump(station_config, output / "meta" / "station_config.json")

    # 3) quantiles.json sanitize
    src_quantiles = json.loads((source / "calibrator" / "quantiles.json").read_text(encoding="utf-8"))
    station_key_counts: Dict[str, int] = {old: 0 for old in old_to_new_active}
    for key in src_quantiles.get("qL", {}):
        has_station, parsed = parse_station_key(key)
        if not has_station:
            continue
        tags = {a: b for a, b in parsed if isinstance(a, str)}
        st = tags.get("S")
        if st in station_key_counts:
            station_key_counts[st] += 1
    missing_station_quantiles = [old for old, n in station_key_counts.items() if n == 0]
    if missing_station_quantiles:
        raise ValueError(
            "Selected demo stations are missing station-specific quantile keys: "
            + ", ".join(missing_station_quantiles)
        )

    out_qL: Dict[str, float] = {}
    out_qU: Dict[str, float] = {}

    for key, vL in src_quantiles.get("qL", {}).items():
        has_station, parsed = parse_station_key(key)
        if has_station:
            new_key = rewrite_station_key(parsed, old_to_new_active)
            if new_key is None:
                continue
        else:
            new_key = key

        vU = src_quantiles.get("qU", {}).get(key)
        if vU is None:
            continue

        out_qL[new_key] = float(vL)
        out_qU[new_key] = float(vU)

    out_quantiles = {
        "alpha": float(src_quantiles.get("alpha", 0.1)),
        "beta": float(src_quantiles.get("beta", 0.9)),
        "min_bin_n": int(src_quantiles.get("min_bin_n", 100)),
        "qL": {k: out_qL[k] for k in sorted(out_qL)},
        "qU": {k: out_qU[k] for k in sorted(out_qU)},
        "bins": src_quantiles.get("bins", {}),
    }
    stable_json_dump(out_quantiles, output / "calibrator" / "quantiles.json")

    # 4) counts.json sanitize
    src_counts = json.loads((source / "calibrator" / "counts.json").read_text(encoding="utf-8"))
    out_counts: Dict[str, Any] = {}

    for key, value in src_counts.items():
        has_station, parsed = parse_station_key(key)
        if has_station:
            new_key = rewrite_station_key(parsed, old_to_new_active)
            if new_key is None:
                continue
        else:
            new_key = key
        out_counts[new_key] = value

    stable_json_dump({k: out_counts[k] for k in sorted(out_counts)}, output / "calibrator" / "counts.json")

    # 5) station_profiles sanitize
    src_profiles = json.loads((source / "calibrator" / "station_profiles.json").read_text(encoding="utf-8"))
    out_profiles = []
    for row in src_profiles:
        if not isinstance(row, dict):
            continue
        old_station = row.get("station")
        if old_station not in old_to_new_active:
            continue
        clean_row = dict(row)
        clean_row["station"] = old_to_new_active[old_station]
        out_profiles.append(clean_row)
    out_profiles = sorted(out_profiles, key=lambda r: r.get("station", ""))
    stable_json_dump(out_profiles, output / "calibrator" / "station_profiles.json")

    # 6) station scalers sanitize
    src_scalers: Dict[str, Any] = joblib.load(source / "transforms" / "station_scalers.joblib")
    out_scalers: Dict[str, Any] = {}
    for old_station, new_station in sorted(old_to_new_active.items(), key=lambda x: x[1]):
        if old_station in src_scalers:
            out_scalers[new_station] = src_scalers[old_station]
    missing_scalers = [s for s in active_new if s not in out_scalers]
    if missing_scalers:
        raise ValueError(f"Missing station scalers for demo stations: {missing_scalers}")
    joblib.dump(out_scalers, output / "transforms" / "station_scalers.joblib")

    # 7,8,9) OHE mapping + dimension + anonymized columns
    src_ohe_cols = json.loads((source / "transforms" / "station_ohe_columns.json").read_text(encoding="utf-8"))
    ohe_dim = len(src_ohe_cols)

    station_ohe_index: Dict[str, int] = {}
    anonymized_ohe_cols: list[str] = []

    for idx, col in enumerate(src_ohe_cols):
        old_station = col.split("st__", 1)[1] if col.startswith("st__") else None
        if old_station and old_station in old_to_new_active:
            new_station = old_to_new_active[old_station]
            station_ohe_index[new_station] = idx
            anonymized_ohe_cols.append(f"st__{new_station}")
        else:
            anonymized_ohe_cols.append(f"st__Hidden_{idx+1:03d}")

    missing_ohe = [s for s in active_new if s not in station_ohe_index]
    if missing_ohe:
        raise ValueError(f"Missing OHE indices for demo stations: {missing_ohe}")

    stable_json_dump(station_ohe_index, output / "transforms" / "station_ohe_index.json")
    stable_json_dump({"ohe_dim": ohe_dim}, output / "transforms" / "ohe_dim.json")
    stable_json_dump(anonymized_ohe_cols, output / "transforms" / "station_ohe_columns.json")

    # 10) empty states dir already created

    # 11) manifest
    manifest = {
        "demo_artifact": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_file": csv_path.name,
        "dataset_sha256": hash_file(csv_path),
        "included_stations": active_new,
        "excluded_stations": sorted(excluded_new),
        "counts": {
            "qL_keys": len(out_qL),
            "qU_keys": len(out_qU),
            "count_keys": len(out_counts),
            "profiles": len(out_profiles),
            "scalers": len(out_scalers),
            "ohe_dim": ohe_dim,
        },
        "note": "Built from a private source artifact and rename map that are intentionally excluded from this public repo.",
    }
    stable_json_dump(manifest, output / "meta" / "demo_manifest.json")

    print(f"[OK] Demo artifact built at: {output}")
    print(f"[OK] Included stations: {', '.join(active_new)}")


if __name__ == "__main__":
    main()
