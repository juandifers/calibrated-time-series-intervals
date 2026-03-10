from __future__ import annotations

from fastapi import APIRouter, Request

from app.config import AppConfig

router = APIRouter()


def _cfg(request: Request) -> AppConfig:
    return request.app.state.cfg


@router.get("/stations")
async def list_stations(request: Request):
    cfg = _cfg(request)
    return [
        {
            "id": station,
            "name": station.replace("_", " "),
            "enabled": True,
        }
        for station in cfg.active_stations
        if station not in cfg.excluded_stations
    ]
