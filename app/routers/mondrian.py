from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.models.service import DemoForecastService

router = APIRouter(prefix="/mondrian", tags=["mondrian"])


class ForecastRequest(BaseModel):
    station: str = Field(..., description="Station identifier")
    forecast_start: str | None = Field(None, description="Forecast start timestamp (ISO8601, UTC)")
    history_hours: int = Field(48, ge=1, le=24 * 60)
    horizon: int | None = Field(None, ge=1, le=96)
    use_runtime_overlays: bool = Field(True, description="Whether to apply runtime calibration overlays")


class ForecastResponse(BaseModel):
    success: bool = True
    station: str
    station_name: str | None = None
    forecast_start: str
    timestamps: list[str]
    predictions: list[float]
    lower_bound: list[float]
    upper_bound: list[float]
    confidence_level: float = 0.90
    horizon: int = 96
    interval_mode: str = "runtime"
    model_version: str = "mondrian_demo_v1"
    historical_timestamps: list[str] | None = None
    historical_values: list[float | None] | None = None
    actual_timestamps: list[str] | None = None
    actual_values: list[float | None] | None = None


class BacktestRequest(BaseModel):
    station: str = Field(..., description="Station identifier")
    backtest_start: str = Field(..., description="Backtest timestamp (ISO8601, UTC)")
    history_hours: int = Field(72, ge=1, le=24 * 60)
    horizon: int | None = Field(None, ge=1, le=96)
    use_runtime_overlays: bool = Field(True, description="Whether to apply runtime calibration overlays")


class BacktestResponse(BaseModel):
    success: bool = True
    station: str
    station_name: str | None = None
    backtest_start: str
    timestamps: list[str]
    predictions: list[float]
    actuals: list[float | None]
    lower_bound: list[float]
    upper_bound: list[float]
    metrics: dict[str, float | int]
    coverage: float
    horizon: int = 96
    interval_mode: str = "runtime"
    model_version: str = "mondrian_demo_v1"


class ModelStatusResponse(BaseModel):
    loaded: bool
    model_type: str = "Mondrian Conformal Prediction"
    mode: str = "single_artifact"
    horizon: int
    num_stations: int
    num_features: int
    calibrator_keys: int
    artifact_path: str
    runtime_state_dir: str
    runtime_state_files: int
    dataset_start: str | None = None
    dataset_end: str | None = None
    allowed_forecast_start_min: str | None = None
    allowed_forecast_start_max: str | None = None
    min_history_hours: int | None = None
    realtime_supported: bool = False
    demo_notice: str | None = None


def _service(request: Request) -> DemoForecastService:
    service = getattr(request.app.state, "forecast_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Forecast service is not initialized",
        )
    return service


@router.post("/forecast", response_model=ForecastResponse)
async def forecast(request_data: ForecastRequest, request: Request):
    service = _service(request)
    try:
        payload = service.forecast(
            station=request_data.station,
            forecast_start=request_data.forecast_start,
            history_hours=request_data.history_hours,
            horizon=request_data.horizon,
            use_runtime_overlays=request_data.use_runtime_overlays,
        )
        payload["station_name"] = request_data.station.replace("_", " ")
        payload.pop("_artifacts", None)
        return ForecastResponse(**payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forecast failed: {exc}",
        ) from exc


@router.post("/backtest", response_model=BacktestResponse)
async def backtest(request_data: BacktestRequest, request: Request):
    service = _service(request)
    try:
        payload = service.backtest(
            station=request_data.station,
            backtest_start=request_data.backtest_start,
            history_hours=request_data.history_hours,
            horizon=request_data.horizon,
            use_runtime_overlays=request_data.use_runtime_overlays,
        )
        payload["station_name"] = request_data.station.replace("_", " ")
        return BacktestResponse(**payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Backtest failed: {exc}",
        ) from exc


@router.get("/status", response_model=ModelStatusResponse)
async def model_status(request: Request):
    service = _service(request)
    status_payload: dict[str, Any] = service.status()
    states_dir = request.app.state.cfg.runtime_states_dir
    status_payload["runtime_state_files"] = len(list(states_dir.glob("*.json")))
    return ModelStatusResponse(**status_payload)


@router.get("/health")
async def health(request: Request):
    _service(request)
    return {"status": "healthy", "horizon": request.app.state.cfg.horizon}
