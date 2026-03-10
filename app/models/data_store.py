from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd

from app.config import AppConfig
from app.models.key_utils import floor_to_grid, normalize_timestamp

logger = logging.getLogger(__name__)


@dataclass
class StationSlice:
    station: str
    data: pd.DataFrame


class DemoDataStore:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._df_long = self._load_dataset(cfg.dataset_path)
        self._station_map = {
            station: g.sort_values("timestamp").reset_index(drop=True)
            for station, g in self._df_long.groupby("station", sort=False)
        }

    def _load_dataset(self, csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

        if "timestamp" not in df.columns:
            raise KeyError("Dataset must contain 'timestamp' column")

        station_cols = [c for c in df.columns if c != "timestamp"]
        missing_active = [s for s in self.cfg.active_stations if s not in station_cols]
        if missing_active:
            raise ValueError(f"Dataset missing active stations: {missing_active}")

        # Keep all columns for transparency, but service only exposes allowlisted active stations.
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])

        df_long = df.melt(id_vars="timestamp", var_name="station", value_name="consumption")
        df_long["station"] = df_long["station"].astype(str)
        df_long = df_long[df_long["station"].isin(self.cfg.active_stations)].copy()

        df_long["consumption"] = pd.to_numeric(df_long["consumption"], errors="coerce")
        df_long = df_long.sort_values(["station", "timestamp"], kind="mergesort").reset_index(drop=True)

        # Set target-compatible columns expected by feature pipeline
        df_long["consumption_clean"] = df_long["consumption"]
        df_long["is_leak"] = 0

        logger.info(
            "Loaded demo dataset %s rows, %s stations, range %s -> %s",
            len(df_long),
            df_long["station"].nunique(),
            df_long["timestamp"].min(),
            df_long["timestamp"].max(),
        )
        return df_long

    @property
    def all_data(self) -> pd.DataFrame:
        return self._df_long

    @property
    def stations(self) -> list[str]:
        return [s for s in sorted(self._station_map) if s in self.cfg.active_stations]

    def latest_timestamp(self) -> pd.Timestamp:
        return self._df_long["timestamp"].max()

    def earliest_timestamp(self) -> pd.Timestamp:
        return self._df_long["timestamp"].min()

    def station_frame(self, station: str) -> pd.DataFrame:
        if station not in self.cfg.active_stations:
            raise ValueError(f"Station '{station}' is not part of the demo scope")
        if station not in self._station_map:
            raise ValueError(f"Station '{station}' not found in dataset")
        return self._station_map[station].copy()

    def station_time_bounds(self, station: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        g = self.station_frame(station)
        if g.empty:
            raise ValueError(f"No data available for station '{station}'")
        return (
            pd.to_datetime(g["timestamp"].min(), utc=True),
            pd.to_datetime(g["timestamp"].max(), utc=True),
        )

    def history_before(self, station: str, end_ts: pd.Timestamp, hours: int) -> pd.DataFrame:
        end_ts = floor_to_grid(normalize_timestamp(end_ts), minutes=self.cfg.step_minutes)
        start_ts = end_ts - timedelta(hours=hours)
        g = self.station_frame(station)
        return g[(g["timestamp"] >= start_ts) & (g["timestamp"] <= end_ts)].copy()

    def actual_window_after(self, station: str, start_ts: pd.Timestamp, horizon: int) -> pd.DataFrame:
        start_ts = floor_to_grid(normalize_timestamp(start_ts), minutes=self.cfg.step_minutes)
        end_ts = start_ts + timedelta(minutes=self.cfg.step_minutes * (horizon - 1))
        g = self.station_frame(station)
        out = g[(g["timestamp"] >= start_ts) & (g["timestamp"] <= end_ts)].copy()
        return out.sort_values("timestamp").reset_index(drop=True)

    def default_forecast_start(self) -> pd.Timestamp:
        # Keep room for actual overlays by defaulting to one horizon before the end.
        return self.latest_timestamp() - timedelta(minutes=self.cfg.step_minutes * self.cfg.horizon)
