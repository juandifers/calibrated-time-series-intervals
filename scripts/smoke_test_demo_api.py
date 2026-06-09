#!/usr/bin/env python3
"""Smoke-test the demo API endpoints."""

from __future__ import annotations

import argparse
import time

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the demo API")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--run-calibration", action="store_true")
    parser.add_argument("--calibration-days", type=int, default=7)
    parser.add_argument("--timeout-sec", type=int, default=300)
    return parser.parse_args()


def _assert(cond: bool, message: str):
    if not cond:
        raise AssertionError(message)


def _get(url: str):
    resp = requests.get(url, timeout=30)
    return resp.status_code, resp.json()


def _post(url: str, payload: dict):
    resp = requests.post(url, json=payload, timeout=60)
    try:
        data = resp.json()
    except Exception:
        data = {"raw": resp.text}
    return resp.status_code, data


def main() -> None:
    args = parse_args()
    base = args.base_url.rstrip("/")

    s_code, stations = _get(f"{base}/stations")
    _assert(s_code == 200, "/stations failed")
    station_ids = [row["id"] for row in stations]
    _assert("Station_6" not in station_ids, "Station_6 must not be exposed by /stations")
    _assert(len(station_ids) == 6, f"Expected 6 active stations, got {len(station_ids)}")
    first_station = station_ids[0]

    status_code, status_payload = _get(f"{base}/api/mondrian/status")
    _assert(status_code == 200, "/api/mondrian/status failed")
    _assert(int(status_payload.get("horizon", -1)) == 96, "Status horizon must be 96")

    f_code, forecast = _post(
        f"{base}/api/mondrian/forecast",
        {"station": first_station, "history_hours": 48, "horizon": 96},
    )
    _assert(f_code == 200, f"Forecast failed for {first_station}: {forecast}")
    _assert(len(forecast.get("predictions", [])) == 96, "Forecast must return 96 predictions")
    _assert(len(forecast.get("lower_bound", [])) == 96, "Forecast lower_bound must return 96 values")
    _assert(len(forecast.get("upper_bound", [])) == 96, "Forecast upper_bound must return 96 values")

    bad_code, bad_forecast = _post(
        f"{base}/api/mondrian/forecast",
        {"station": "Station_6", "history_hours": 48, "horizon": 96},
    )
    _assert(bad_code >= 400, "Forecast with Station_6 must fail")
    _assert("Station_6" in str(bad_forecast), "Station_6 failure should be explicit")

    replay_start = status_payload.get("allowed_forecast_start_max")
    _assert(replay_start, "Status payload must expose allowed_forecast_start_max")

    reset_code, reset_payload = _post(
        f"{base}/api/calibration/reset",
        {"stations": [first_station]},
    )
    _assert(reset_code == 200, f"Reset request failed: {reset_payload}")

    cmp_code, cmp_payload = _post(
        f"{base}/api/calibration/replay-compare",
        {
            "station": first_station,
            "replay_start": replay_start,
            "history_hours": 48,
            "days": int(args.calibration_days),
        },
    )
    _assert(cmp_code == 200, f"Replay comparison failed: {cmp_payload}")
    _assert(cmp_payload.get("station") == first_station, "Replay comparison station mismatch")
    _assert(cmp_payload.get("before", {}).get("interval_mode") == "base", "Before comparison must use base intervals")
    _assert(cmp_payload.get("after", {}).get("interval_mode") == "runtime", "After comparison must use runtime intervals")

    if args.run_calibration:
        c_code, c_payload = _post(
            f"{base}/api/calibration/calibrate",
            {"stations": [first_station], "days": int(args.calibration_days)},
        )
        _assert(c_code == 200, f"Calibration request failed: {c_payload}")
        job_id = c_payload.get("job_id")
        _assert(job_id, "Calibration response missing job_id")

        deadline = time.time() + args.timeout_sec
        while time.time() < deadline:
            j_code, j_payload = _get(f"{base}/api/calibration/jobs/{job_id}")
            _assert(j_code == 200, f"Failed to poll calibration job {job_id}: {j_payload}")
            state = j_payload.get("status")
            if state in {"succeeded", "failed", "cancelled"}:
                _assert(state == "succeeded", f"Calibration job ended in non-success state: {state}")
                break
            time.sleep(2)
        else:
            raise TimeoutError(f"Calibration job {job_id} did not finish within {args.timeout_sec}s")

    reset_code, reset_payload = _post(
        f"{base}/api/calibration/reset",
        {"stations": [first_station]},
    )
    _assert(reset_code == 200, f"Final reset request failed: {reset_payload}")

    print("[OK] API smoke tests passed")


if __name__ == "__main__":
    main()
