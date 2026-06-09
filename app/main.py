from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.wsgi import WSGIMiddleware

from app.config import load_config
from app.models.bundle import load_demo_bundle
from app.models.calibration import CalibrationSettings, ManualCalibrationEngine
from app.models.data_store import DemoDataStore
from app.models.intervals import IntervalResolver
from app.models.jobs import JobManager
from app.models.service import DemoForecastService
from app.routers import calibration as calibration_router
from app.routers import mondrian as mondrian_router
from app.routers import stations as stations_router
from dashboard.app import create_dash_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    cfg.runtime_states_dir.mkdir(parents=True, exist_ok=True)
    cfg.runtime_jobs_path.parent.mkdir(parents=True, exist_ok=True)
    app.state.cfg = cfg

    bundle = load_demo_bundle(cfg.artifact_path)
    artifact_h = int(bundle.meta.get("horizon", cfg.horizon))
    if artifact_h != cfg.horizon:
        raise ValueError(
            f"Demo service requires fixed horizon={cfg.horizon}, but artifact reports horizon={artifact_h}"
        )

    data_store = DemoDataStore(cfg)
    interval_resolver = IntervalResolver(bundle=bundle, runtime_states_dir=cfg.runtime_states_dir)
    forecast_service = DemoForecastService(
        cfg=cfg,
        bundle=bundle,
        data_store=data_store,
        intervals=interval_resolver,
    )
    calibration_engine = ManualCalibrationEngine(
        cfg=cfg,
        bundle=bundle,
        data_store=data_store,
        intervals=interval_resolver,
        settings=CalibrationSettings(
            horizon=cfg.horizon,
            stride=4,
            beta=0.90,
            min_sht_n=6,
            min_sh_n=100,
        ),
    )
    job_manager = JobManager(cfg.runtime_jobs_path)

    app.state.bundle = bundle
    app.state.data_store = data_store
    app.state.interval_resolver = interval_resolver
    app.state.forecast_service = forecast_service
    app.state.calibration_engine = calibration_engine
    app.state.job_manager = job_manager

    logger.info(
        "Demo service initialized (stations=%s, horizon=%s, artifact=%s)",
        len(cfg.active_stations),
        cfg.horizon,
        cfg.artifact_path,
    )

    yield


app = FastAPI(
    title="Calibrated Time-Series Intervals — Demo API",
    description="Public demo service for single-artifact Mondrian forecasting and manual calibration",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(mondrian_router.router, prefix="/api", tags=["mondrian"])
app.include_router(calibration_router.router, prefix="/api", tags=["calibration"])
app.include_router(stations_router.router, tags=["stations"])

dash_app = create_dash_app()
app.mount("/dashboard", WSGIMiddleware(dash_app.server))


@app.get("/")
async def root():
    return {
        "message": "Calibrated Time-Series Intervals — Demo API",
        "docs": "/docs",
        "dashboard": "/dashboard",
    }


@app.get("/health")
async def health():
    cfg = getattr(app.state, "cfg", None)
    horizon = getattr(cfg, "horizon", 96)
    return {"status": "healthy", "horizon": horizon}
