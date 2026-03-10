from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from app.config import AppConfig
from app.models.bundle import DemoBundle
from app.models.data_store import DemoDataStore
from app.models.features import build_last_window
from app.models.intervals import IntervalResolver
from app.models.key_utils import floor_to_grid, forecast_timestamps, normalize_timestamp, timestamp_to_tod_bin

logger = logging.getLogger(__name__)


@dataclass
class ForecastArtifacts:
    x_window_shape: tuple[int, int]
    interval_sources: list[str]
    feature_rows: int


class DemoForecastService:
    def __init__(
        self,
        cfg: AppConfig,
        bundle: DemoBundle,
        data_store: DemoDataStore,
        intervals: IntervalResolver,
    ):
        self.cfg = cfg
        self.bundle = bundle
        self.data_store = data_store
        self.intervals = intervals

    def min_history_hours(self) -> int:
        # Need at least one full input window for model features.
        hours = (self.cfg.in_length * self.cfg.step_minutes) / 60.0
        return max(1, int(np.ceil(hours)))

    def forecast_start_bounds(self, station: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        station_min, station_max = self.data_store.station_time_bounds(station)
        min_start = floor_to_grid(
            station_min + timedelta(minutes=self.cfg.in_length * self.cfg.step_minutes),
            minutes=self.cfg.step_minutes,
        )
        # Keep full horizon inside dataset so demo can always show actual overlays.
        max_start = floor_to_grid(
            station_max - timedelta(minutes=self.cfg.step_minutes * (self.cfg.horizon - 1)),
            minutes=self.cfg.step_minutes,
        )
        return min_start, max_start

    def _validate_demo_scope(self, station: str, start_ts: pd.Timestamp, history_hours: int) -> None:
        min_h = self.min_history_hours()
        if int(history_hours) < min_h:
            raise ValueError(
                f"history_hours must be >= {min_h} for this demo "
                f"(got {history_hours})."
            )

        min_start, max_start = self.forecast_start_bounds(station)
        if start_ts < min_start or start_ts > max_start:
            raise ValueError(
                "This public demo only supports historical replay within dataset bounds. "
                f"Allowed forecast_start for {station}: "
                f"{min_start.isoformat()} to {max_start.isoformat()} (UTC). "
                f"Received: {start_ts.isoformat()}."
            )

    def resolve_horizon(self, requested: int | None) -> int:
        if requested is None:
            return self.cfg.horizon
        if int(requested) != self.cfg.horizon:
            raise ValueError(f"Horizon is fixed to {self.cfg.horizon} for the public demo")
        return self.cfg.horizon

    def assert_station_supported(self, station: str) -> None:
        if station in self.cfg.excluded_stations:
            raise ValueError(f"Station '{station}' is excluded from this demo")
        if station not in self.cfg.active_stations:
            raise ValueError(f"Station '{station}' is not part of the active demo station scope")

    def _inverse_scale(self, station: str, arr: np.ndarray) -> np.ndarray:
        y = np.asarray(arr, dtype=np.float32).reshape(-1)
        scaler = self.bundle.station_scalers.get(station)
        if scaler is None:
            return y.astype(np.float32, copy=False)
        out = scaler.inverse_transform(y.reshape(-1, 1)).ravel()
        return np.asarray(out, dtype=np.float32)

    def _predict_scaled(self, x_window: np.ndarray) -> np.ndarray:
        yhat = np.asarray(self.bundle.model.predict(x_window), dtype=np.float32)
        if yhat.ndim == 1:
            yhat = yhat.reshape(1, -1)
        if yhat.shape[1] != self.cfg.horizon:
            raise RuntimeError(
                f"Model output horizon mismatch: expected {self.cfg.horizon}, got {yhat.shape[1]}"
            )
        return yhat

    def forecast(
        self,
        station: str,
        forecast_start: pd.Timestamp | str | None = None,
        history_hours: int = 24,
        horizon: int | None = None,
        use_runtime_overlays: bool = True,
    ) -> dict[str, Any]:
        self.assert_station_supported(station)
        horizon = self.resolve_horizon(horizon)
        if horizon != self.cfg.horizon:
            raise RuntimeError("Unexpected horizon drift")
        history_hours = max(1, int(history_hours))

        if forecast_start is None:
            start_ts = self.data_store.default_forecast_start()
        else:
            start_ts = normalize_timestamp(forecast_start)
        start_ts = floor_to_grid(start_ts, minutes=self.cfg.step_minutes)
        self._validate_demo_scope(station=station, start_ts=start_ts, history_hours=history_hours)

        station_df = self.data_store.station_frame(station)
        x_window, _last_ts, feature_frame = build_last_window(
            bundle=self.bundle,
            cfg=self.cfg,
            station=station,
            station_df=station_df,
            forecast_start=start_ts,
            history_hours=history_hours,
        )

        yhat_scaled = self._predict_scaled(x_window)[0]
        yhat = self._inverse_scale(station, yhat_scaled)

        ts_out = forecast_timestamps(start_ts, horizon=self.cfg.horizon, step_minutes=self.cfg.step_minutes)
        lower: list[float] = []
        upper: list[float] = []
        interval_sources: list[str] = []

        for h, ts_h in enumerate(ts_out):
            t_bin = timestamp_to_tod_bin(ts_h, tz=self.cfg.tz, bins=self.cfg.horizon)
            lo, up, src = self.intervals.interval(
                station,
                h,
                t_bin,
                float(yhat[h]),
                use_runtime_overlays=use_runtime_overlays,
            )
            lower.append(float(lo))
            upper.append(float(up))
            interval_sources.append(src)

        hist_end = start_ts - timedelta(minutes=self.cfg.step_minutes)
        hist = self.data_store.history_before(station=station, end_ts=hist_end, hours=history_hours)
        actual = self.data_store.actual_window_after(station=station, start_ts=start_ts, horizon=self.cfg.horizon)

        artifacts = ForecastArtifacts(
            x_window_shape=tuple(int(v) for v in x_window.shape),
            interval_sources=interval_sources,
            feature_rows=int(len(feature_frame)),
        )

        return {
            "success": True,
            "station": station,
            "forecast_start": start_ts.isoformat(),
            "timestamps": [ts.isoformat() for ts in ts_out],
            "predictions": [float(v) for v in yhat.tolist()],
            "lower_bound": lower,
            "upper_bound": upper,
            "confidence_level": 0.90,
            "horizon": self.cfg.horizon,
            "interval_mode": "runtime" if use_runtime_overlays else "base",
            "model_version": "mondrian_demo_v1",
            "historical_timestamps": [pd.Timestamp(ts).isoformat() for ts in hist["timestamp"].tolist()],
            "historical_values": [float(v) if pd.notna(v) else None for v in hist["consumption"].tolist()],
            "actual_timestamps": [pd.Timestamp(ts).isoformat() for ts in actual["timestamp"].tolist()],
            "actual_values": [float(v) if pd.notna(v) else None for v in actual["consumption"].tolist()],
            "_artifacts": {
                "x_window_shape": list(artifacts.x_window_shape),
                "feature_rows": artifacts.feature_rows,
                "interval_sources": artifacts.interval_sources,
            },
        }

    def backtest(
        self,
        station: str,
        backtest_start: pd.Timestamp | str,
        history_hours: int = 48,
        horizon: int | None = None,
        use_runtime_overlays: bool = True,
    ) -> dict[str, Any]:
        self.assert_station_supported(station)
        horizon = self.resolve_horizon(horizon)
        start_ts = floor_to_grid(normalize_timestamp(backtest_start), minutes=self.cfg.step_minutes)

        fc = self.forecast(
            station=station,
            forecast_start=start_ts,
            history_hours=history_hours,
            horizon=horizon,
            use_runtime_overlays=use_runtime_overlays,
        )
        ts = pd.to_datetime(fc["timestamps"], utc=True, errors="coerce")
        yhat = np.asarray(fc["predictions"], dtype=np.float32)
        lower = np.asarray(fc["lower_bound"], dtype=np.float32)
        upper = np.asarray(fc["upper_bound"], dtype=np.float32)

        actual_df = self.data_store.actual_window_after(station=station, start_ts=start_ts, horizon=self.cfg.horizon)
        actual_map = {
            pd.Timestamp(row["timestamp"]).isoformat(): float(row["consumption"])
            for _, row in actual_df.iterrows()
            if pd.notna(row["consumption"])
        }
        actuals = np.array([actual_map.get(pd.Timestamp(t).isoformat(), np.nan) for t in ts], dtype=np.float32)
        valid = np.isfinite(actuals)
        if not valid.any():
            raise ValueError("No actual observations available for backtest horizon")

        mae = float(np.mean(np.abs(yhat[valid] - actuals[valid])))
        rmse = float(np.sqrt(np.mean((yhat[valid] - actuals[valid]) ** 2)))
        coverage = float(np.mean((actuals[valid] >= lower[valid]) & (actuals[valid] <= upper[valid])))
        mean_width = float(np.mean((upper - lower)[valid]))

        return {
            "success": True,
            "station": station,
            "backtest_start": start_ts.isoformat(),
            "timestamps": fc["timestamps"],
            "predictions": [float(v) for v in yhat.tolist()],
            "actuals": [float(v) if np.isfinite(v) else None for v in actuals.tolist()],
            "lower_bound": [float(v) for v in lower.tolist()],
            "upper_bound": [float(v) for v in upper.tolist()],
            "metrics": {
                "mae": mae,
                "rmse": rmse,
                "mean_interval_width": mean_width,
                "n_valid_points": int(valid.sum()),
                "n_total_points": int(len(valid)),
            },
            "coverage": coverage,
            "horizon": self.cfg.horizon,
            "interval_mode": "runtime" if use_runtime_overlays else "base",
            "model_version": "mondrian_demo_v1",
        }

    def status(self) -> dict[str, Any]:
        stations = [s for s in self.cfg.active_stations if s not in self.cfg.excluded_stations]
        if stations:
            bounds = [self.forecast_start_bounds(s) for s in stations]
            min_start = max(b[0] for b in bounds)
            max_start = min(b[1] for b in bounds)
        else:
            ts = self.data_store.latest_timestamp()
            min_start = ts
            max_start = ts

        return {
            "loaded": True,
            "model_type": "Mondrian Conformal Prediction",
            "mode": "single_artifact",
            "horizon": self.cfg.horizon,
            "num_stations": len(self.cfg.active_stations),
            "num_features": len(self.bundle.meta.get("features", [])),
            "calibrator_keys": len(self.bundle.qL),
            "artifact_path": str(self.cfg.artifact_path),
            "runtime_state_dir": str(self.cfg.runtime_states_dir),
            "dataset_start": self.data_store.earliest_timestamp().isoformat(),
            "dataset_end": self.data_store.latest_timestamp().isoformat(),
            "allowed_forecast_start_min": min_start.isoformat(),
            "allowed_forecast_start_max": max_start.isoformat(),
            "min_history_hours": self.min_history_hours(),
            "realtime_supported": False,
            "demo_notice": "Historical replay only: forecast_start must stay within the dataset time range.",
        }
