from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from app.models.bundle import DemoBundle
from app.models.key_utils import make_h_key, make_h_tb_key, make_station_key, make_station_parent_key, make_tb_key

logger = logging.getLogger(__name__)


@dataclass
class IntervalLookup:
    qL: float
    qU: float
    source: str
    units: str = "original"


class IntervalResolver:
    def __init__(self, bundle: DemoBundle, runtime_states_dir: Path):
        self.bundle = bundle
        self.runtime_states_dir = Path(runtime_states_dir)
        self.runtime_states_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_mtime: dict[str, float] = {}
        self._lock = RLock()

    @staticmethod
    def _abs_width(v: Any) -> float:
        try:
            return float(abs(float(v)))
        except Exception:
            return 0.0

    @staticmethod
    def _normalize_units(v: Any, default: str = "original") -> str:
        units = str(v or default).strip().lower()
        if units not in {"original", "scaled"}:
            return default
        return units

    def _station_scale_factor(self, station: str) -> float:
        scaler = self.bundle.station_scalers.get(station)
        if scaler is None:
            return 1.0
        try:
            scale = float(scaler.scale_[0] if hasattr(scaler.scale_, "__len__") else scaler.scale_)
        except Exception:
            return 1.0
        if not np.isfinite(scale) or scale <= 0:
            return 1.0
        return scale

    def _state_file(self, station: str) -> Path:
        return self.runtime_states_dir / f"{station}.json"

    def load_runtime_state(self, station: str) -> dict[str, Any]:
        with self._lock:
            p = self._state_file(station)
            if not p.exists():
                return {
                    "station": station,
                    "keys": {},
                    "updated_at": None,
                    "quantile_units": "original",
                }

            mtime = p.stat().st_mtime
            if station in self._cache and self._cache_mtime.get(station) == mtime:
                return self._cache[station]

            data = json.loads(p.read_text(encoding="utf-8"))
            if "keys" not in data or not isinstance(data["keys"], dict):
                data["keys"] = {}
            data["quantile_units"] = self._normalize_units(data.get("quantile_units"), default="original")
            self._cache[station] = data
            self._cache_mtime[station] = mtime
            return data

    def save_runtime_state(self, station: str, data: dict[str, Any]) -> None:
        with self._lock:
            p = self._state_file(station)
            data = dict(data)
            data["updated_at"] = datetime.now(timezone.utc).isoformat()
            data["quantile_units"] = self._normalize_units(data.get("quantile_units"), default="original")
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(p)
            self._cache[station] = data
            self._cache_mtime[station] = p.stat().st_mtime

    def clear_runtime_state(self, station: str) -> bool:
        with self._lock:
            p = self._state_file(station)
            existed = p.exists()
            if existed:
                p.unlink()
            self._cache.pop(station, None)
            self._cache_mtime.pop(station, None)
            return existed

    def summarize_runtime_state(self, station: str) -> dict[str, Any]:
        state = self.load_runtime_state(station)
        keys = state.get("keys", {})
        return {
            "station": station,
            "updated_at": state.get("updated_at"),
            "num_keys": len(keys),
            "sample_keys": list(keys.keys())[:10],
        }

    def _lookup_runtime(self, station: str, h: int, t_bin: int) -> IntervalLookup | None:
        state = self.load_runtime_state(station)
        keys = state.get("keys", {})
        state_units = self._normalize_units(state.get("quantile_units"), default="original")

        k_sht = make_station_key(station, h, t_bin)
        if k_sht in keys:
            entry = keys[k_sht]
            return IntervalLookup(
                qL=self._abs_width(entry.get("qL", 0.0)),
                qU=self._abs_width(entry.get("qU", 0.0)),
                source="runtime:SHT",
                units=self._normalize_units(entry.get("units"), default=state_units),
            )

        k_sh = make_station_parent_key(station, h)
        if k_sh in keys:
            entry = keys[k_sh]
            return IntervalLookup(
                qL=self._abs_width(entry.get("qL", 0.0)),
                qU=self._abs_width(entry.get("qU", 0.0)),
                source="runtime:SH",
                units=self._normalize_units(entry.get("units"), default=state_units),
            )

        return None

    def _lookup_base_station(self, station: str, h: int, t_bin: int) -> IntervalLookup | None:
        key = make_station_key(station, h, t_bin)
        if key in self.bundle.qL and key in self.bundle.qU:
            return IntervalLookup(
                qL=self._abs_width(self.bundle.qL[key]),
                qU=self._abs_width(self.bundle.qU[key]),
                source="base:SHT",
                units="scaled",
            )
        return None

    def _lookup_pooled(self, h: int, t_bin: int) -> IntervalLookup | None:
        key_ht = make_h_tb_key(h, t_bin)
        if key_ht in self.bundle.bins:
            b = self.bundle.bins[key_ht]
            return IntervalLookup(
                self._abs_width(b.get("qL", 0.0)),
                self._abs_width(b.get("qU", 0.0)),
                "pooled:H-T",
                units="scaled",
            )

        key_h = make_h_key(h)
        key_t = make_tb_key(t_bin)

        # blend h/t if both present
        if key_h in self.bundle.bins and key_t in self.bundle.bins:
            bh = self.bundle.bins[key_h]
            bt = self.bundle.bins[key_t]
            return IntervalLookup(
                qL=(self._abs_width(bh.get("qL", 0.0)) + self._abs_width(bt.get("qL", 0.0))) / 2.0,
                qU=(self._abs_width(bh.get("qU", 0.0)) + self._abs_width(bt.get("qU", 0.0))) / 2.0,
                source="pooled:blend(H,T)",
                units="scaled",
            )

        if key_h in self.bundle.bins:
            b = self.bundle.bins[key_h]
            return IntervalLookup(
                self._abs_width(b.get("qL", 0.0)),
                self._abs_width(b.get("qU", 0.0)),
                "pooled:H",
                units="scaled",
            )

        if key_t in self.bundle.bins:
            b = self.bundle.bins[key_t]
            return IntervalLookup(
                self._abs_width(b.get("qL", 0.0)),
                self._abs_width(b.get("qU", 0.0)),
                "pooled:T",
                units="scaled",
            )

        if "GLOBAL" in self.bundle.bins:
            b = self.bundle.bins["GLOBAL"]
            return IntervalLookup(
                self._abs_width(b.get("qL", 0.0)),
                self._abs_width(b.get("qU", 0.0)),
                "pooled:GLOBAL",
                units="scaled",
            )

        # hard fallback from station q maps median
        if self.bundle.qL and self.bundle.qU:
            return IntervalLookup(
                qL=float(np.median(np.abs(list(self.bundle.qL.values())))),
                qU=float(np.median(np.abs(list(self.bundle.qU.values())))),
                source="fallback:median(q)",
                units="scaled",
            )

        return IntervalLookup(qL=1.0, qU=1.0, source="fallback:default", units="scaled")

    def _to_original_widths(self, station: str, lookup: IntervalLookup) -> tuple[float, float]:
        qL = max(0.0, float(lookup.qL))
        qU = max(0.0, float(lookup.qU))
        units = self._normalize_units(lookup.units, default="original")
        if units == "scaled":
            scale = self._station_scale_factor(station)
            qL *= scale
            qU *= scale
        return qL, qU

    def interval(
        self,
        station: str,
        h: int,
        t_bin: int,
        yhat: float,
        use_runtime_overlays: bool = True,
    ) -> tuple[float, float, str]:
        runtime_hit = self._lookup_runtime(station, h, t_bin) if use_runtime_overlays else None
        if runtime_hit:
            qL, qU = self._to_original_widths(station, runtime_hit)
            return yhat - qL, yhat + qU, runtime_hit.source

        station_hit = self._lookup_base_station(station, h, t_bin)
        if station_hit:
            qL, qU = self._to_original_widths(station, station_hit)
            return yhat - qL, yhat + qU, station_hit.source

        pooled_hit = self._lookup_pooled(h, t_bin)
        if pooled_hit is None:
            pooled_hit = IntervalLookup(qL=1.0, qU=1.0, source="fallback:default", units="scaled")
        qL, qU = self._to_original_widths(station, pooled_hit)
        return yhat - qL, yhat + qU, pooled_hit.source
