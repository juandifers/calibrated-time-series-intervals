from __future__ import annotations

from collections import Counter
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.models.calibration import ManualCalibrationEngine
from app.models.intervals import IntervalResolver
from app.models.jobs import JobManager
from app.models.service import DemoForecastService

router = APIRouter(prefix="/calibration", tags=["calibration"])


class CalibrationRequest(BaseModel):
    stations: list[str] | Literal["all"] = Field(
        ...,
        description="Stations to calibrate, or 'all' for full demo scope",
    )
    days: int = Field(..., ge=1, le=730)
    reference_time: str | None = Field(
        None,
        description="Optional replay cutoff timestamp (ISO8601, UTC). Only historical data before this timestamp is used.",
    )


class CalibrationResetRequest(BaseModel):
    stations: list[str] | Literal["all"] = Field(
        "all",
        description="Stations whose runtime calibration overlays should be removed, or 'all'.",
    )


class ReplayCompareRequest(BaseModel):
    station: str = Field(..., description="Station identifier")
    replay_start: str = Field(..., description="Replay start timestamp (ISO8601, UTC)")
    history_hours: int = Field(48, ge=1, le=24 * 60)
    days: int = Field(..., ge=1, le=730, description="Trailing calibration window in days")


class ReplayCompareResponse(BaseModel):
    station: str
    replay_start: str
    history_hours: int
    days: int
    note: str
    calibration_result: dict[str, Any]
    runtime_state: dict[str, Any]
    before: dict[str, Any]
    after: dict[str, Any]
    comparison: dict[str, Any]


class CalibrationJobResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    progress: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    cancelled: bool = False


def _jobs(request: Request) -> JobManager:
    jobs = getattr(request.app.state, "job_manager", None)
    if jobs is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Calibration job manager is not initialized",
        )
    return jobs


def _calibration_engine(request: Request) -> ManualCalibrationEngine:
    engine = getattr(request.app.state, "calibration_engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Calibration engine is not initialized",
        )
    return engine


def _resolver(request: Request) -> IntervalResolver:
    resolver = getattr(request.app.state, "interval_resolver", None)
    if resolver is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Interval resolver is not initialized",
        )
    return resolver


def _service(request: Request) -> DemoForecastService:
    service = getattr(request.app.state, "forecast_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Forecast service is not initialized",
        )
    return service


def _resolve_target_stations(request: Request, stations: list[str] | Literal["all"]) -> list[str]:
    cfg = request.app.state.cfg
    allowed = [s for s in cfg.active_stations if s not in cfg.excluded_stations]
    if stations == "all":
        return allowed
    if not stations:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Stations list cannot be empty")
    unknown = [s for s in stations if s not in allowed]
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported station(s) in calibration request: {unknown}",
        )
    # Preserve input order and remove duplicates.
    deduped = list(dict.fromkeys(stations))
    return deduped


def _forecast_summary(payload: dict[str, Any]) -> dict[str, Any]:
    timestamps = payload.get("timestamps") or []
    lower = payload.get("lower_bound") or []
    upper = payload.get("upper_bound") or []
    actual_ts = payload.get("actual_timestamps") or []
    actual_vals = payload.get("actual_values") or []

    actual_map = {
        str(ts): float(v)
        for ts, v in zip(actual_ts, actual_vals)
        if v is not None
    }

    widths: list[float] = []
    hits = 0
    total = 0
    for ts, lo, up in zip(timestamps, lower, upper):
        try:
            lo_f = float(lo)
            up_f = float(up)
        except Exception:
            continue
        widths.append(up_f - lo_f)
        if ts in actual_map:
            y = actual_map[ts]
            total += 1
            if lo_f <= y <= up_f:
                hits += 1

    coverage = float(hits / total) if total else None
    width_avg = float(sum(widths) / len(widths)) if widths else None
    width_delta = None
    if payload.get("_artifacts"):
        width_delta = dict(Counter(payload["_artifacts"].get("interval_sources") or []))

    return {
        "coverage": coverage,
        "coverage_hits": hits,
        "coverage_total": total,
        "mean_interval_width": width_avg,
        "interval_source_counts": width_delta or {},
    }


@router.post("/calibrate", response_model=CalibrationJobResponse)
async def calibrate(payload: CalibrationRequest, request: Request):
    jobs = _jobs(request)
    engine = _calibration_engine(request)
    target_stations = _resolve_target_stations(request, payload.stations)

    request_dict = {
        "stations": target_stations,
        "days": int(payload.days),
        "reference_time": payload.reference_time,
    }
    job_id = jobs.create_job(request=request_dict)

    def _runner() -> dict[str, Any]:
        return engine.calibrate(
            stations=target_stations,
            days=int(payload.days),
            reference_time=payload.reference_time,
            job_id=job_id,
            jobs=jobs,
        )

    jobs.run_background(job_id=job_id, fn=_runner)
    return CalibrationJobResponse(**jobs.get_job(job_id))


@router.post("/reset")
async def reset_calibration(payload: CalibrationResetRequest, request: Request):
    resolver = _resolver(request)
    target_stations = _resolve_target_stations(request, payload.stations)
    cleared = [station for station in target_stations if resolver.clear_runtime_state(station)]
    return {
        "stations_requested": target_stations,
        "stations_cleared": cleared,
        "count_cleared": len(cleared),
    }


@router.post("/replay-compare", response_model=ReplayCompareResponse)
async def replay_compare(payload: ReplayCompareRequest, request: Request):
    service = _service(request)
    engine = _calibration_engine(request)
    resolver = _resolver(request)
    _resolve_target_stations(request, [payload.station])

    try:
        before = service.forecast(
            station=payload.station,
            forecast_start=payload.replay_start,
            history_hours=int(payload.history_hours),
            use_runtime_overlays=False,
        )
        calibration_result = engine.calibrate_station(
            station=payload.station,
            days=int(payload.days),
            reference_time=payload.replay_start,
        )
        after = service.forecast(
            station=payload.station,
            forecast_start=payload.replay_start,
            history_hours=int(payload.history_hours),
            use_runtime_overlays=True,
        )
        runtime_state = resolver.summarize_runtime_state(payload.station)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Replay calibration comparison failed: {exc}",
        ) from exc

    before_summary = _forecast_summary(before)
    after_summary = _forecast_summary(after)
    before_width = before_summary.get("mean_interval_width")
    after_width = after_summary.get("mean_interval_width")
    before_cov = before_summary.get("coverage")
    after_cov = after_summary.get("coverage")

    comparison = {
        "before": before_summary,
        "after": after_summary,
        "delta_mean_interval_width": (
            float(after_width - before_width)
            if before_width is not None and after_width is not None
            else None
        ),
        "delta_coverage": (
            float(after_cov - before_cov)
            if before_cov is not None and after_cov is not None
            else None
        ),
    }

    return ReplayCompareResponse(
        station=payload.station,
        replay_start=payload.replay_start,
        history_hours=int(payload.history_hours),
        days=int(payload.days),
        note="The comparison uses the immutable base artifact for 'before' and the persisted runtime overlay for 'after'. Use /api/calibration/reset to clear the overlay.",
        calibration_result=calibration_result,
        runtime_state=runtime_state,
        before=before,
        after=after,
        comparison=comparison,
    )


@router.get("/jobs", response_model=list[CalibrationJobResponse])
async def list_jobs(request: Request):
    jobs = _jobs(request)
    return [CalibrationJobResponse(**rec) for rec in jobs.list_jobs()]


@router.get("/jobs/{job_id}", response_model=CalibrationJobResponse)
async def get_job(job_id: str, request: Request):
    jobs = _jobs(request)
    try:
        return CalibrationJobResponse(**jobs.get_job(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found") from exc


@router.post("/jobs/{job_id}/cancel", response_model=CalibrationJobResponse)
async def cancel_job(job_id: str, request: Request):
    jobs = _jobs(request)
    try:
        return CalibrationJobResponse(**jobs.cancel_job(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found") from exc


@router.get("/state/{station_id}")
async def station_state(station_id: str, request: Request):
    cfg = request.app.state.cfg
    allowed = [s for s in cfg.active_stations if s not in cfg.excluded_stations]
    if station_id not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported station '{station_id}' for calibration state",
        )

    resolver = _resolver(request)
    state = resolver.load_runtime_state(station_id)
    keys = state.get("keys", {})
    kind_counts: dict[str, int] = {}
    for v in keys.values():
        kind = str(v.get("kind", "unknown"))
        kind_counts[kind] = kind_counts.get(kind, 0) + 1

    return {
        "station_id": station_id,
        "updated_at": state.get("updated_at"),
        "num_keys": len(keys),
        "kind_counts": kind_counts,
        "sample_keys": list(keys.keys())[:20],
        "summary": state.get("summary", {}),
        "settings": {
            "days": state.get("days"),
            "horizon": state.get("horizon"),
            "stride": state.get("stride"),
            "beta": state.get("beta"),
            "min_sht_n": state.get("min_sht_n"),
            "min_sh_n": state.get("min_sh_n"),
        },
    }


@router.get("/config")
async def calibration_config(request: Request):
    engine = _calibration_engine(request)
    return {
        "horizon": engine.settings.horizon,
        "stride": engine.settings.stride,
        "beta": engine.settings.beta,
        "min_sht_n": engine.settings.min_sht_n,
        "min_sh_n": engine.settings.min_sh_n,
        "runtime_state_dir": str(request.app.state.cfg.runtime_states_dir),
    }
