from __future__ import annotations

import ast
from datetime import timedelta
from typing import Any

import pandas as pd


def normalize_timestamp(ts: pd.Timestamp | str | None) -> pd.Timestamp:
    if ts is None:
        out = pd.Timestamp.utcnow().tz_localize("UTC") if pd.Timestamp.utcnow().tzinfo is None else pd.Timestamp.utcnow()
        return out.tz_convert("UTC")
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        out = out.tz_localize("UTC")
    else:
        out = out.tz_convert("UTC")
    return out


def floor_to_grid(ts: pd.Timestamp, minutes: int = 15) -> pd.Timestamp:
    ts = normalize_timestamp(ts)
    minute = (ts.minute // minutes) * minutes
    return ts.replace(minute=minute, second=0, microsecond=0)


def timestamp_to_tod_bin(ts: pd.Timestamp, tz: str = "Europe/Madrid", bins: int = 96) -> int:
    ts = normalize_timestamp(ts)
    ts_local = ts.tz_convert(tz)
    mins = ts_local.hour * 60 + ts_local.minute
    idx = int((mins * bins) // 1440)
    return max(0, min(bins - 1, idx))


def make_station_key(station: str, h: int, t_bin: int) -> str:
    return str((("S", station), ("H", int(h)), ("T", int(t_bin))))


def make_station_parent_key(station: str, h: int) -> str:
    return str((("S", station), ("H", int(h))))


def make_h_tb_key(h: int, t_bin: int) -> str:
    return f"h={int(h)}|tb={int(t_bin)}"


def make_h_key(h: int) -> str:
    return f"h={int(h)}"


def make_tb_key(t_bin: int) -> str:
    return f"tb={int(t_bin)}"


def parse_key_safe(key: str) -> Any:
    try:
        return ast.literal_eval(key)
    except Exception:
        return key


def extract_station_from_key(key: str) -> str | None:
    obj = parse_key_safe(key)
    if isinstance(obj, tuple) and obj and isinstance(obj[0], tuple):
        tags = {a: b for a, b in obj if isinstance(a, str)}
        return tags.get("S")
    return None


def forecast_timestamps(forecast_start: pd.Timestamp, horizon: int, step_minutes: int) -> list[pd.Timestamp]:
    start = normalize_timestamp(forecast_start)
    return [start + timedelta(minutes=step_minutes * h) for h in range(horizon)]
