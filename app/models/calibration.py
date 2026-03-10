from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from app.config import AppConfig
from app.models.bundle import DemoBundle
from app.models.data_store import DemoDataStore
from app.models.features import build_feature_frame
from app.models.intervals import IntervalResolver
from app.models.jobs import JobManager
from app.models.key_utils import floor_to_grid, make_station_key, make_station_parent_key, normalize_timestamp, timestamp_to_tod_bin

logger = logging.getLogger(__name__)


@dataclass
class CalibrationSettings:
    horizon: int = 96
    stride: int = 4
    beta: float = 0.90
    min_sht_n: int = 6
    min_sh_n: int = 100


class ManualCalibrationEngine:
    def __init__(
        self,
        cfg: AppConfig,
        bundle: DemoBundle,
        data_store: DemoDataStore,
        intervals: IntervalResolver,
        settings: CalibrationSettings | None = None,
    ):
        self.cfg = cfg
        self.bundle = bundle
        self.data_store = data_store
        self.intervals = intervals
        self.settings = settings or CalibrationSettings()

        if self.settings.horizon != cfg.horizon:
            raise ValueError(
                f"Calibration settings horizon ({self.settings.horizon}) must match fixed service horizon ({cfg.horizon})"
            )

    def _inverse_scale(self, station: str, arr: np.ndarray) -> np.ndarray:
        y = np.asarray(arr, dtype=np.float32).reshape(-1)
        scaler = self.bundle.station_scalers.get(station)
        if scaler is None:
            return y
        out = scaler.inverse_transform(y.reshape(-1, 1)).ravel()
        return np.asarray(out, dtype=np.float32)

    def _candidate_start_indices(self, frame: pd.DataFrame, days: int) -> np.ndarray:
        n = len(frame)
        if n < self.cfg.in_length + self.cfg.horizon:
            return np.array([], dtype=np.int32)

        ts = pd.to_datetime(frame["timestamp"], utc=True)
        end_ts = ts.iloc[-1]
        min_start = end_ts - pd.Timedelta(days=int(days))

        i_min = self.cfg.in_length
        i_max = n - self.cfg.horizon
        if i_max < i_min:
            return np.array([], dtype=np.int32)

        idx = np.arange(i_min, i_max + 1, self.settings.stride, dtype=np.int32)
        keep = ts.iloc[idx] >= min_start
        return idx[keep.to_numpy(dtype=bool)]

    def calibrate_station(
        self,
        station: str,
        days: int,
        reference_time: pd.Timestamp | str | None = None,
    ) -> dict[str, Any]:
        if station in self.cfg.excluded_stations:
            raise ValueError(f"Station '{station}' is excluded from this demo")
        if station not in self.cfg.active_stations:
            raise ValueError(f"Station '{station}' is not supported in this demo")

        station_df = self.data_store.station_frame(station)
        if station_df.empty:
            raise ValueError(f"No data found for station '{station}'")

        latest_ts = pd.to_datetime(station_df["timestamp"].max(), utc=True)
        if reference_time is None:
            forecast_start = latest_ts + timedelta(minutes=self.cfg.step_minutes)
        else:
            forecast_start = floor_to_grid(
                normalize_timestamp(reference_time),
                minutes=self.cfg.step_minutes,
            )
        frame = build_feature_frame(
            bundle=self.bundle,
            cfg=self.cfg,
            station=station,
            station_df=station_df,
            forecast_start=forecast_start,
        )

        feature_cols = list(self.bundle.meta.get("features", []))
        if not feature_cols:
            raise ValueError("Artifact metadata is missing feature column definitions")
        for col in feature_cols:
            if col not in frame.columns:
                frame[col] = 0.0

        idx = self._candidate_start_indices(frame, days=days)
        if len(idx) == 0:
            raise ValueError(
                f"Insufficient usable windows for station '{station}' in the last {days} day(s)"
            )

        ohe_vec = np.zeros(self.bundle.ohe_dim, dtype=np.float32)
        ohe_idx = self.bundle.station_ohe_index.get(station)
        if ohe_idx is None:
            raise ValueError(f"No OHE index configured for station '{station}'")
        if not (0 <= ohe_idx < self.bundle.ohe_dim):
            raise ValueError(f"Invalid OHE index for station '{station}': {ohe_idx}")
        ohe_vec[ohe_idx] = 1.0
        ohe_flat = np.repeat(ohe_vec.reshape(1, -1), self.cfg.in_length, axis=0).reshape(-1)

        feat_vals = frame[feature_cols].to_numpy(dtype=np.float32, copy=False)
        target_scaled = pd.to_numeric(frame["cons_scaled"], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        ts_vals = pd.to_datetime(frame["timestamp"], utc=True).to_numpy()

        x_list: list[np.ndarray] = []
        y_true_scaled: list[np.ndarray] = []
        ts_future: list[np.ndarray] = []

        for i in idx:
            x_base = feat_vals[i - self.cfg.in_length : i]
            y_slice = target_scaled[i : i + self.cfg.horizon]
            t_slice = ts_vals[i : i + self.cfg.horizon]

            if x_base.shape[0] != self.cfg.in_length or y_slice.shape[0] != self.cfg.horizon:
                continue
            if not np.isfinite(x_base).all() or not np.isfinite(y_slice).all():
                continue

            x_flat = np.concatenate([x_base.reshape(-1), ohe_flat], axis=0).astype(np.float32, copy=False)
            x_list.append(x_flat)
            y_true_scaled.append(y_slice.astype(np.float32, copy=False))
            ts_future.append(t_slice)

        if not x_list:
            raise ValueError(f"No valid calibration windows after filtering for station '{station}'")

        X = np.vstack(x_list).astype(np.float32, copy=False)
        Y_true_scaled = np.vstack(y_true_scaled).astype(np.float32, copy=False)
        T_future = np.vstack(ts_future)

        Y_hat_scaled = np.asarray(self.bundle.model.predict(X), dtype=np.float32)
        if Y_hat_scaled.ndim == 1:
            Y_hat_scaled = Y_hat_scaled.reshape(1, -1)
        if Y_hat_scaled.shape != Y_true_scaled.shape:
            raise RuntimeError(
                f"Prediction shape mismatch for station '{station}': "
                f"pred={Y_hat_scaled.shape}, true={Y_true_scaled.shape}"
            )

        Y_hat = self._inverse_scale(station, Y_hat_scaled.reshape(-1)).reshape(Y_hat_scaled.shape)
        Y_true = self._inverse_scale(station, Y_true_scaled.reshape(-1)).reshape(Y_true_scaled.shape)

        eL = np.maximum(Y_hat - Y_true, 0.0)
        eU = np.maximum(Y_true - Y_hat, 0.0)

        store_sht_l: dict[str, list[float]] = defaultdict(list)
        store_sht_u: dict[str, list[float]] = defaultdict(list)
        store_sh_l: dict[str, list[float]] = defaultdict(list)
        store_sh_u: dict[str, list[float]] = defaultdict(list)

        n_windows = Y_hat.shape[0]
        for w in range(n_windows):
            for h in range(self.cfg.horizon):
                ts_h = normalize_timestamp(pd.Timestamp(T_future[w, h]))
                t_bin = timestamp_to_tod_bin(ts_h, tz=self.cfg.tz, bins=self.cfg.horizon)

                key_sht = make_station_key(station, h, t_bin)
                key_sh = make_station_parent_key(station, h)

                el = float(eL[w, h])
                eu = float(eU[w, h])
                if np.isfinite(el):
                    store_sht_l[key_sht].append(el)
                    store_sh_l[key_sh].append(el)
                if np.isfinite(eu):
                    store_sht_u[key_sht].append(eu)
                    store_sh_u[key_sh].append(eu)

        keys: dict[str, dict[str, Any]] = {}
        beta = float(self.settings.beta)

        n_sht = 0
        for k in sorted(store_sht_l):
            l_vals = store_sht_l[k]
            u_vals = store_sht_u.get(k, [])
            n = min(len(l_vals), len(u_vals))
            if n < self.settings.min_sht_n:
                continue
            ql = float(np.quantile(np.asarray(l_vals[:n], dtype=np.float32), beta))
            qu = float(np.quantile(np.asarray(u_vals[:n], dtype=np.float32), beta))
            keys[k] = {"qL": max(0.0, ql), "qU": max(0.0, qu), "n": int(n), "kind": "SHT"}
            n_sht += 1

        n_sh = 0
        for k in sorted(store_sh_l):
            l_vals = store_sh_l[k]
            u_vals = store_sh_u.get(k, [])
            n = min(len(l_vals), len(u_vals))
            if n < self.settings.min_sh_n:
                continue
            ql = float(np.quantile(np.asarray(l_vals[:n], dtype=np.float32), beta))
            qu = float(np.quantile(np.asarray(u_vals[:n], dtype=np.float32), beta))
            keys[k] = {"qL": max(0.0, ql), "qU": max(0.0, qu), "n": int(n), "kind": "SH"}
            n_sh += 1

        if not keys:
            raise ValueError(
                f"Calibration produced no usable overlay keys for station '{station}'. "
                f"Try increasing `days`."
            )

        coverage_total = 0
        coverage_hits = 0
        for w in range(n_windows):
            for h in range(self.cfg.horizon):
                ts_h = normalize_timestamp(pd.Timestamp(T_future[w, h]))
                t_bin = timestamp_to_tod_bin(ts_h, tz=self.cfg.tz, bins=self.cfg.horizon)
                key_sht = make_station_key(station, h, t_bin)
                key_sh = make_station_parent_key(station, h)
                q = keys.get(key_sht) or keys.get(key_sh)
                if q is None:
                    continue
                lo = float(Y_hat[w, h] - float(q["qL"]))
                up = float(Y_hat[w, h] + float(q["qU"]))
                y = float(Y_true[w, h])
                coverage_total += 1
                if lo <= y <= up:
                    coverage_hits += 1

        coverage = None
        if coverage_total > 0:
            coverage = float(coverage_hits / coverage_total)

        state = {
            "station": station,
            "days": int(days),
            "horizon": int(self.cfg.horizon),
            "stride": int(self.settings.stride),
            "beta": float(self.settings.beta),
            "min_sht_n": int(self.settings.min_sht_n),
            "min_sh_n": int(self.settings.min_sh_n),
            # Runtime overlays are learned on descaled targets (original units).
            "quantile_units": "original",
            "keys": keys,
            "summary": {
                "windows_scored": int(n_windows),
                "points_scored": int(n_windows * self.cfg.horizon),
                "updated_keys": int(len(keys)),
                "updated_sht_keys": int(n_sht),
                "updated_sh_keys": int(n_sh),
                "empirical_coverage": coverage,
                "train_start": normalize_timestamp(pd.Timestamp(T_future[0, 0])).isoformat(),
                "train_end": normalize_timestamp(pd.Timestamp(T_future[-1, -1])).isoformat(),
                "reference_time": forecast_start.isoformat(),
            },
        }
        self.intervals.save_runtime_state(station, state)

        return {
            "station": station,
            "days": int(days),
            "reference_time": forecast_start.isoformat(),
            "windows_scored": int(n_windows),
            "updated_keys": int(len(keys)),
            "updated_sht_keys": int(n_sht),
            "updated_sh_keys": int(n_sh),
            "empirical_coverage": coverage,
        }

    def calibrate(
        self,
        stations: list[str],
        days: int,
        reference_time: pd.Timestamp | str | None = None,
        job_id: str | None = None,
        jobs: JobManager | None = None,
    ) -> dict[str, Any]:
        target_stations = [s for s in stations if s in self.cfg.active_stations and s not in self.cfg.excluded_stations]
        if not target_stations:
            raise ValueError("No valid stations selected for calibration")

        results: list[dict[str, Any]] = []
        completed = 0

        for station in target_stations:
            if job_id and jobs and jobs.is_cancelled(job_id):
                logger.info("Calibration job %s cancelled before station %s", job_id, station)
                break

            if job_id and jobs:
                jobs.update_progress(
                    job_id,
                    current_station=station,
                    stations_completed=completed,
                    total_stations=len(target_stations),
                    current_stage="calibrating_station",
                    progress_percent=float((completed / max(1, len(target_stations))) * 100.0),
                )

            result = self.calibrate_station(
                station=station,
                days=days,
                reference_time=reference_time,
            )
            results.append(result)
            completed += 1

            if job_id and jobs:
                jobs.update_progress(
                    job_id,
                    current_station=station,
                    stations_completed=completed,
                    total_stations=len(target_stations),
                    current_stage="station_complete",
                    progress_percent=float((completed / max(1, len(target_stations))) * 100.0),
                )

        if job_id and jobs:
            jobs.update_progress(
                job_id,
                current_station=None,
                stations_completed=completed,
                total_stations=len(target_stations),
                current_stage="finished",
                progress_percent=100.0 if completed == len(target_stations) else float(
                    (completed / max(1, len(target_stations))) * 100.0
                ),
            )

        return {
            "stations_requested": len(target_stations),
            "stations_calibrated": completed,
            "days": int(days),
            "reference_time": normalize_timestamp(reference_time).isoformat() if reference_time is not None else None,
            "results": results,
            "total_keys_updated": int(sum(r.get("updated_keys", 0) for r in results)),
        }
