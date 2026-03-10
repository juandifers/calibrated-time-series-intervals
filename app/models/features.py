from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from app.config import AppConfig
from app.models.bundle import DemoBundle
from app.models.key_utils import floor_to_grid, normalize_timestamp

logger = logging.getLogger(__name__)


def _scale_station_series(bundle: DemoBundle, station: str, s: pd.Series) -> np.ndarray:
    scaler = bundle.station_scalers.get(station)
    values = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float32)
    if scaler is None:
        return values
    scaled = scaler.transform(values.reshape(-1, 1)).ravel().astype(np.float32)
    return scaled


def add_calendar_features(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    out = df.sort_values("timestamp").copy()
    ts_local = pd.to_datetime(out["timestamp"], utc=True).dt.tz_convert(tz)

    out["dow_local"] = ts_local.dt.weekday.astype(np.int16)
    out["hour_local"] = ts_local.dt.hour.astype(np.int16)
    out["minute_local"] = ts_local.dt.minute.astype(np.int16)

    minutes = out["hour_local"] * 60 + out["minute_local"]
    out["tod_sin"] = np.sin(2 * np.pi * minutes / (24 * 60)).astype(np.float32)
    out["tod_cos"] = np.cos(2 * np.pi * minutes / (24 * 60)).astype(np.float32)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow_local"] / 7.0).astype(np.float32)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow_local"] / 7.0).astype(np.float32)
    return out


def add_target_features_in_time(df: pd.DataFrame, value_col: str = "cons_scaled") -> pd.DataFrame:
    out = df.sort_values("timestamp").copy()
    t = pd.to_numeric(out[value_col], errors="coerce").astype(np.float32)

    out["lag_96"] = t.shift(96)
    out["diff_96"] = t - out["lag_96"]

    r4 = t.rolling(window=4, min_periods=1)
    out["mean_1h"] = r4.mean()

    r24 = t.rolling(window=24, min_periods=1)
    out["med_6h"] = r24.median()
    out["std_6h"] = r24.std()

    r96 = t.rolling(window=96, min_periods=1)
    out["max_24h"] = r96.max()
    out["std_24h"] = r96.std()

    out["lag96_dow_interact"] = out["lag_96"] * out["dow_cos"]

    for col in [
        "mean_1h",
        "med_6h",
        "std_6h",
        "lag_96",
        "diff_96",
        "lag96_dow_interact",
        "dow_sin",
        "dow_cos",
        "max_24h",
        "std_24h",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype(np.float32)

    return out


def build_feature_frame(
    bundle: DemoBundle,
    cfg: AppConfig,
    station: str,
    station_df: pd.DataFrame,
    forecast_start: pd.Timestamp,
) -> pd.DataFrame:
    """Build full feature frame up to just before forecast_start."""
    forecast_start = floor_to_grid(normalize_timestamp(forecast_start), minutes=cfg.step_minutes)
    cutoff = forecast_start - timedelta(minutes=cfg.step_minutes)

    g = station_df[station_df["timestamp"] <= cutoff].copy()
    g = g.sort_values("timestamp").reset_index(drop=True)
    if g.empty:
        raise ValueError(f"No history available before forecast_start for {station}")

    # Ensure regular grid per station
    g = g.set_index("timestamp").asfreq(f"{cfg.step_minutes}min")
    g["station"] = station

    # Fill short gaps conservatively for inference
    cons = pd.to_numeric(g["consumption_clean"], errors="coerce")
    g["consumption_clean"] = cons.interpolate(method="time", limit=2, limit_direction="both")
    g["consumption_clean"] = g["consumption_clean"].ffill().bfill()

    g = g.reset_index()

    g["cons_scaled"] = _scale_station_series(bundle, station, g["consumption_clean"])
    g = add_calendar_features(g, tz=cfg.tz)
    g = add_target_features_in_time(g, value_col="cons_scaled")
    g["is_bad"] = 0
    return g


def _station_ohe_vector(bundle: DemoBundle, station: str) -> np.ndarray:
    vec = np.zeros(bundle.ohe_dim, dtype=np.float32)
    idx = bundle.station_ohe_index.get(station)
    if idx is None:
        raise ValueError(f"No OHE index configured for station {station}")
    if not (0 <= idx < bundle.ohe_dim):
        raise ValueError(f"Invalid OHE index for station {station}: {idx}")
    vec[idx] = 1.0
    return vec


def build_last_window(
    bundle: DemoBundle,
    cfg: AppConfig,
    station: str,
    station_df: pd.DataFrame,
    forecast_start: pd.Timestamp,
    history_hours: int,
) -> tuple[np.ndarray, pd.Timestamp, pd.DataFrame]:
    frame = build_feature_frame(bundle, cfg, station, station_df, forecast_start)

    # Optional trimming to speed up computation when very long histories are passed
    if history_hours > 0:
        floor_ts = floor_to_grid(normalize_timestamp(forecast_start), minutes=cfg.step_minutes) - timedelta(hours=history_hours)
        frame = frame[frame["timestamp"] >= floor_ts].copy()
        if frame.empty:
            raise ValueError("History window produced empty feature frame")

    if len(frame) < cfg.in_length:
        raise ValueError(f"Insufficient rows for window: {len(frame)} < {cfg.in_length}")

    base_features = list(bundle.meta.get("features", []))
    if not base_features:
        raise ValueError("Artifact metadata missing 'features'")

    for col in base_features:
        if col not in frame.columns:
            frame[col] = 0.0

    window_df = frame.iloc[-cfg.in_length:].copy()
    x_base = window_df[base_features].to_numpy(dtype=np.float32, copy=False)
    ohe_vec = _station_ohe_vector(bundle, station)
    x_ohe = np.repeat(ohe_vec.reshape(1, -1), cfg.in_length, axis=0)

    x_window = np.hstack([x_base, x_ohe]).reshape(1, -1).astype(np.float32)
    last_ts = pd.to_datetime(window_df["timestamp"].iloc[-1], utc=True)
    return x_window, last_ts, frame
