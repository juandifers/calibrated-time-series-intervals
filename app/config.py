from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    dataset_path: Path
    artifact_path: Path
    runtime_states_dir: Path
    runtime_jobs_path: Path
    horizon: int
    in_length: int
    step_minutes: int
    tz: str
    active_stations: tuple[str, ...]
    excluded_stations: tuple[str, ...]


def _resolve_project_root() -> Path:
    # app/config.py -> repo root is two levels up. Resolve by location,
    # not by directory name, so a fresh clone works regardless of folder name.
    root = Path(__file__).resolve().parents[1]
    if not (root / "df_stationsv3.csv").exists():
        raise FileNotFoundError(
            f"Could not locate project root: expected df_stationsv3.csv under {root}"
        )
    return root


def load_config() -> AppConfig:
    root = _resolve_project_root()
    artifact = root / "mondrian_artifacts_demo"
    station_cfg_path = artifact / "meta" / "station_config.json"

    if station_cfg_path.exists():
        import json

        station_cfg = json.loads(station_cfg_path.read_text(encoding="utf-8"))
        active = tuple(station_cfg.get("active_stations", []))
        excluded = tuple(station_cfg.get("excluded_stations", []))
    else:
        # fallback before demo artifact is built
        active = (
            "Station_1",
            "Station_2",
            "Station_3",
            "Station_4",
            "Station_7",
            "Station_8",
        )
        excluded = ("Station_6",)

    return AppConfig(
        project_root=root,
        dataset_path=root / "df_stationsv3.csv",
        artifact_path=artifact,
        runtime_states_dir=root / "runtime" / "calibration_states",
        runtime_jobs_path=root / "runtime" / "calibration_jobs.json",
        horizon=96,
        in_length=96,
        step_minutes=15,
        tz="Europe/Madrid",
        active_stations=active,
        excluded_stations=excluded,
    )
