#!/usr/bin/env python3
"""Compute real backtest metrics from the demo replay path and render a figure.

Reuses the exact app objects wired in app/main.py at startup (no model
reimplementation). Requires the model to be fetched first:

    python scripts/fetch_demo_model.py
    python scripts/compute_demo_metrics.py

Outputs:
  - docs/results.md           Markdown table (per-station + overall)
  - docs/img/example_forecast.png   Representative replay figure
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

# Mirror the wiring in app/main.py startup_event exactly.
from app.config import load_config  # noqa: E402
from app.models.bundle import load_demo_bundle  # noqa: E402
from app.models.data_store import DemoDataStore  # noqa: E402
from app.models.intervals import IntervalResolver  # noqa: E402
from app.models.service import DemoForecastService  # noqa: E402

NOMINAL_COVERAGE = 0.90
N_WINDOWS = 10
HISTORY_HOURS = 48


def build_service() -> tuple[DemoForecastService, object]:
    cfg = load_config()
    bundle = load_demo_bundle(cfg.artifact_path)
    data_store = DemoDataStore(cfg)
    intervals = IntervalResolver(bundle=bundle, runtime_states_dir=cfg.runtime_states_dir)
    service = DemoForecastService(
        cfg=cfg,
        bundle=bundle,
        data_store=data_store,
        intervals=intervals,
    )
    return service, cfg


def sample_cutoffs(min_start: pd.Timestamp, max_start: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    """Evenly spaced cutoffs strictly inside (min_start, max_start)."""
    if max_start <= min_start:
        return []
    span = max_start - min_start
    cutoffs: list[pd.Timestamp] = []
    seen: set[pd.Timestamp] = set()
    for i in range(1, n + 1):
        frac = i / (n + 1)  # 1/(n+1) .. n/(n+1): strictly interior
        # Snap to the 15-min replay grid; sub-grid timestamps would desync the
        # forecast horizon from the on-grid actuals and yield no observations.
        ts = (min_start + span * frac).floor("15min")
        if ts <= min_start or ts >= max_start or ts in seen:
            continue
        seen.add(ts)
        cutoffs.append(ts)
    return cutoffs


def aggregate(rows: list[dict]) -> dict | None:
    """Weight per-window metrics by n_valid_points; combine RMSE in MSE space."""
    rows = [r for r in rows if r["n_valid"] > 0]
    if not rows:
        return None
    total_n = sum(r["n_valid"] for r in rows)
    coverage = sum(r["coverage"] * r["n_valid"] for r in rows) / total_n
    mae = sum(r["mae"] * r["n_valid"] for r in rows) / total_n
    rmse = math.sqrt(sum((r["rmse"] ** 2) * r["n_valid"] for r in rows) / total_n)
    width = sum(r["width"] * r["n_valid"] for r in rows) / total_n
    return {
        "coverage": coverage,
        "mae": mae,
        "rmse": rmse,
        "width": width,
        "n_valid": total_n,
        "n_windows": len(rows),
    }


def collect_station(service: DemoForecastService, station: str) -> list[dict]:
    min_start, max_start = service.forecast_start_bounds(station)
    rows: list[dict] = []
    for cutoff in sample_cutoffs(min_start, max_start, N_WINDOWS):
        try:
            bt = service.backtest(
                station,
                cutoff,
                history_hours=HISTORY_HOURS,
                use_runtime_overlays=False,
            )
        except ValueError as exc:
            if "No actual observations" in str(exc):
                continue
            raise
        m = bt["metrics"]
        rows.append(
            {
                "cutoff": cutoff,
                "coverage": float(bt["coverage"]),
                "mae": float(m["mae"]),
                "rmse": float(m["rmse"]),
                "width": float(m["mean_interval_width"]),
                "n_valid": int(m["n_valid_points"]),
            }
        )
    return rows


def write_results_md(path: Path, per_station: dict[str, dict], overall: dict) -> None:
    lines = [
        "# Backtest Results",
        "",
        "| Station | Coverage (target 0.90) | MAE | RMSE | Mean interval width | Replay windows |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for station in sorted(per_station):
        a = per_station[station]
        lines.append(
            f"| {station} | {a['coverage']:.3f} | {a['mae']:.3f} | {a['rmse']:.3f} | "
            f"{a['width']:.3f} | {a['n_windows']} |"
        )
    lines.append(
        f"| **Overall** | **{overall['coverage']:.3f}** | **{overall['mae']:.3f}** | "
        f"**{overall['rmse']:.3f}** | **{overall['width']:.3f}** | **{overall['n_windows']}** |"
    )
    lines.append("")
    lines.append(
        f"_Empirical coverage vs. a nominal {NOMINAL_COVERAGE:.0%} target, aggregated over "
        f"{overall['n_windows']} historical replay windows (base intervals, no runtime overlays), "
        f"weighted by the number of valid forecast points._"
    )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def render_figure(service: DemoForecastService, station: str, out_path: Path) -> pd.Timestamp:
    min_start, max_start = service.forecast_start_bounds(station)
    cutoffs = sample_cutoffs(min_start, max_start, N_WINDOWS)
    cutoff = cutoffs[len(cutoffs) // 2] if cutoffs else min_start

    fc = service.forecast(
        station=station,
        forecast_start=cutoff,
        history_hours=HISTORY_HOURS,
        use_runtime_overlays=False,
    )

    hist_t = pd.to_datetime(fc["historical_timestamps"], utc=True)
    hist_v = pd.to_numeric(pd.Series(fc["historical_values"]), errors="coerce")
    fc_t = pd.to_datetime(fc["timestamps"], utc=True)
    pred = np.asarray(fc["predictions"], dtype=float)
    lower = np.asarray(fc["lower_bound"], dtype=float)
    upper = np.asarray(fc["upper_bound"], dtype=float)
    actual = pd.to_numeric(pd.Series(fc["actual_values"]), errors="coerce")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(hist_t, hist_v, color="#486581", lw=1.8, label="History")
    ax.fill_between(
        fc_t, lower, upper, color="#0f4c81", alpha=0.18,
        label="90% prediction interval",
    )
    ax.plot(fc_t, pred, color="#0f4c81", lw=2.4, label="Prediction")
    ax.plot(fc_t, actual.to_numpy(dtype=float), color="#d1495b", lw=1.8, label="Actual")
    ax.axvline(pd.Timestamp(fc["forecast_start"]), color="#9aa5b1", ls="--", lw=1.0)

    ax.set_title(f"Historical replay forecast — {station} (cutoff {fc['forecast_start']})")
    ax.set_xlabel("Timestamp (UTC)")
    ax.set_ylabel("Flow / consumption")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return cutoff


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute demo backtest metrics and render a figure")
    parser.add_argument("--results", type=Path, default=repo_root / "docs" / "results.md")
    parser.add_argument("--figure", type=Path, default=repo_root / "docs" / "img" / "example_forecast.png")
    args = parser.parse_args()

    service, cfg = build_service()
    stations = [s for s in cfg.active_stations if s not in cfg.excluded_stations]

    per_station: dict[str, dict] = {}
    all_rows: list[dict] = []
    for station in stations:
        rows = collect_station(service, station)
        agg = aggregate(rows)
        if agg is None:
            print(f"[warn] {station}: no usable replay windows, skipped")
            continue
        per_station[station] = agg
        all_rows.extend(rows)
        print(
            f"[ok] {station}: coverage={agg['coverage']:.3f} mae={agg['mae']:.3f} "
            f"rmse={agg['rmse']:.3f} width={agg['width']:.3f} windows={agg['n_windows']}"
        )

    overall = aggregate(all_rows)
    if overall is None:
        raise SystemExit("No usable backtest windows across any station.")

    write_results_md(args.results, per_station, overall)
    fig_station = stations[0]
    cutoff = render_figure(service, fig_station, args.figure)

    print("\n=== OVERALL ===")
    print(
        f"coverage={overall['coverage']:.3f} mae={overall['mae']:.3f} "
        f"rmse={overall['rmse']:.3f} width={overall['width']:.3f} "
        f"windows={overall['n_windows']}"
    )
    print(f"results -> {args.results}")
    print(f"figure  -> {args.figure} (station {fig_station}, cutoff {cutoff})")


if __name__ == "__main__":
    main()
