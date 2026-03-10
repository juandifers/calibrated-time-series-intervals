from __future__ import annotations

import json
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class JobRecord:
    job_id: str
    request: dict[str, Any]
    status: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    cancelled: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "request": self.request,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "cancelled": self.cancelled,
        }


class JobManager:
    def __init__(self, jobs_path: Path):
        self.jobs_path = Path(jobs_path)
        self.jobs_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="demo-calibration")
        self._jobs: dict[str, JobRecord] = {}
        self._futures: dict[str, Future] = {}
        self._load()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _save(self) -> None:
        payload = {job_id: rec.as_dict() for job_id, rec in self._jobs.items()}
        tmp = self.jobs_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.jobs_path)

    def _load(self) -> None:
        if not self.jobs_path.exists():
            return
        try:
            payload = json.loads(self.jobs_path.read_text(encoding="utf-8"))
            for job_id, raw in payload.items():
                self._jobs[job_id] = JobRecord(**raw)
        except Exception as exc:
            logger.warning("Failed to load persisted jobs from %s: %s", self.jobs_path, exc)

    def create_job(self, request: dict[str, Any]) -> str:
        with self._lock:
            job_id = uuid4().hex
            rec = JobRecord(
                job_id=job_id,
                request=request,
                status="queued",
                created_at=self._now(),
                progress={
                    "stations_completed": 0,
                    "total_stations": 0,
                    "current_station": None,
                    "current_stage": "queued",
                    "progress_percent": 0.0,
                },
            )
            self._jobs[job_id] = rec
            self._save()
            return job_id

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._jobs[k].as_dict() for k in sorted(self._jobs, key=lambda j: self._jobs[j].created_at, reverse=True)]

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id].as_dict()

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                raise KeyError(job_id)
            rec.cancelled = True
            if rec.status in {"queued", "running"}:
                rec.status = "cancelled"
                rec.finished_at = self._now()
            self._save()
            future = self._futures.get(job_id)
            if future and not future.done():
                future.cancel()
            return rec.as_dict()

    def update_job(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return
            for k, v in fields.items():
                setattr(rec, k, v)
            self._save()

    def update_progress(self, job_id: str, **progress_fields: Any) -> None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return
            rec.progress.update(progress_fields)
            self._save()

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            rec = self._jobs.get(job_id)
            return bool(rec.cancelled) if rec else True

    def run_background(self, job_id: str, fn: Callable[[], dict[str, Any]]) -> None:
        def _runner() -> None:
            self.update_job(job_id, status="running", started_at=self._now())
            try:
                result = fn()
                if self.is_cancelled(job_id):
                    self.update_job(job_id, status="cancelled", finished_at=self._now())
                    return
                self.update_job(
                    job_id,
                    status="succeeded",
                    finished_at=self._now(),
                    result=result,
                    error=None,
                )
            except Exception as exc:
                logger.exception("Background job %s failed", job_id)
                if self.is_cancelled(job_id):
                    self.update_job(job_id, status="cancelled", finished_at=self._now(), error=str(exc))
                else:
                    self.update_job(job_id, status="failed", finished_at=self._now(), error=str(exc))

        future = self._executor.submit(_runner)
        with self._lock:
            self._futures[job_id] = future
