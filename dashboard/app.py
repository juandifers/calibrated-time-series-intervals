from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
from flask import request as flask_request


DISPLAY_TZ = "Europe/Madrid"


def _resolve_api_base_url() -> str:
    env_url = (os.getenv("DASH_API_BASE_URL") or "").strip()
    if env_url:
        return env_url.rstrip("/")
    try:
        return flask_request.host_url.rstrip("/")
    except Exception:
        return "http://localhost:8000"


def _call_api(
    path: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout_sec: int | float | None = None,
) -> Any:
    base = _resolve_api_base_url()
    url = f"{base}{path}"
    timeout = float(timeout_sec) if timeout_sec is not None else (120.0 if method.upper() == "POST" else 30.0)
    if method.upper() == "POST":
        resp = requests.post(url, json=payload or {}, timeout=timeout)
    else:
        resp = requests.get(url, timeout=timeout)
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail")
        except Exception:
            detail = resp.text
        raise RuntimeError(f"{resp.status_code} {path}: {detail}")
    return resp.json()


def _clean_xy(timestamps: list[Any], values: list[Any]) -> tuple[list[pd.Timestamp], list[float]]:
    n = min(len(timestamps or []), len(values or []))
    if n == 0:
        return [], []

    df = pd.DataFrame(
        {
            "t": pd.to_datetime((timestamps or [])[:n], utc=True, errors="coerce"),
            "y": pd.to_numeric((values or [])[:n], errors="coerce"),
        }
    ).dropna(subset=["t", "y"])
    if df.empty:
        return [], []

    ts_local = pd.to_datetime(df["t"], utc=True).dt.tz_convert(DISPLAY_TZ).dt.tz_localize(None)
    return ts_local.tolist(), [float(v) for v in df["y"].tolist()]


def _clean_interval(
    timestamps: list[Any],
    lower: list[Any],
    upper: list[Any],
) -> tuple[list[pd.Timestamp], list[float], list[float]]:
    n = min(len(timestamps or []), len(lower or []), len(upper or []))
    if n == 0:
        return [], [], []

    df = pd.DataFrame(
        {
            "t": pd.to_datetime((timestamps or [])[:n], utc=True, errors="coerce"),
            "lower": pd.to_numeric((lower or [])[:n], errors="coerce"),
            "upper": pd.to_numeric((upper or [])[:n], errors="coerce"),
        }
    ).dropna(subset=["t", "lower", "upper"])
    if df.empty:
        return [], [], []

    low = pd.to_numeric(df["lower"], errors="coerce")
    up = pd.to_numeric(df["upper"], errors="coerce")
    lo = low.where(low <= up, up)
    hi = up.where(low <= up, low)

    ts_local = pd.to_datetime(df["t"], utc=True).dt.tz_convert(DISPLAY_TZ).dt.tz_localize(None)
    return ts_local.tolist(), [float(v) for v in lo.tolist()], [float(v) for v in hi.tolist()]


def _compute_picp_with_tolerance(
    interval_ts: list[Any],
    lower: list[Any],
    upper: list[Any],
    actual_ts: list[Any],
    actual_vals: list[Any],
    tolerance_minutes: float = 7.5,
) -> tuple[float | None, int, int]:
    if not interval_ts or not lower or not upper or not actual_ts or not actual_vals:
        return None, 0, 0

    df_int = pd.DataFrame(
        {
            "t_int": pd.to_datetime(interval_ts, utc=True, errors="coerce"),
            "lower": pd.to_numeric(lower, errors="coerce"),
            "upper": pd.to_numeric(upper, errors="coerce"),
        }
    ).dropna(subset=["t_int", "lower", "upper"])

    df_act = pd.DataFrame(
        {
            "t_act": pd.to_datetime(actual_ts, utc=True, errors="coerce"),
            "y": pd.to_numeric(actual_vals, errors="coerce"),
        }
    ).dropna(subset=["t_act", "y"])

    if df_int.empty or df_act.empty:
        return None, 0, 0

    df_int = df_int.sort_values("t_int")
    df_act = df_act.sort_values("t_act")

    tol = pd.Timedelta(minutes=tolerance_minutes)
    merged = pd.merge_asof(
        df_act,
        df_int,
        left_on="t_act",
        right_on="t_int",
        direction="nearest",
        tolerance=tol,
    ).dropna(subset=["lower", "upper"])

    if merged.empty:
        return None, 0, 0

    y = merged["y"].to_numpy()
    lower_arr = merged["lower"].to_numpy()
    upper_arr = merged["upper"].to_numpy()
    inside = (lower_arr <= y) & (y <= upper_arr)
    hits = int(inside.sum())
    total = int(len(y))
    if total <= 0:
        return None, 0, 0
    return float(hits) / float(total), hits, total


def _coverage_from_forecast(response: dict[str, Any]) -> str:
    picp, hits, total = _compute_picp_with_tolerance(
        interval_ts=response.get("timestamps") or [],
        lower=response.get("lower_bound") or [],
        upper=response.get("upper_bound") or [],
        actual_ts=response.get("actual_timestamps") or [],
        actual_vals=response.get("actual_values") or [],
    )
    if picp is None:
        return "PICP: not available (no aligned actuals)"
    return f"PICP (coverage): {picp:.1%} ({hits}/{total})"


def _interval_width_summary(response: dict[str, Any]) -> str:
    lo = np.asarray(pd.to_numeric(response.get("lower_bound") or [], errors="coerce"), dtype=float)
    up = np.asarray(pd.to_numeric(response.get("upper_bound") or [], errors="coerce"), dtype=float)
    if lo.size == 0 or up.size == 0:
        return ""

    n = min(lo.size, up.size)
    widths = up[:n] - lo[:n]
    valid = widths[np.isfinite(widths)]
    if valid.size == 0:
        return ""

    return (
        f"Interval width avg={float(np.mean(valid)):.2f}, "
        f"p90={float(np.quantile(valid, 0.90)):.2f}, "
        f"max={float(np.max(valid)):.2f}"
    )


def _build_demo_scope_notice(status_payload: dict[str, Any]) -> tuple[str | None, str | None, str | None, int]:
    tz = ZoneInfo(DISPLAY_TZ)
    min_ts = pd.to_datetime(status_payload.get("allowed_forecast_start_min"), utc=True, errors="coerce")
    max_ts = pd.to_datetime(status_payload.get("allowed_forecast_start_max"), utc=True, errors="coerce")

    if pd.isna(min_ts) or pd.isna(max_ts):
        today = datetime.now().date().isoformat()
        return today, today, today, 0

    min_local = min_ts.tz_convert(tz)
    max_local = max_ts.tz_convert(tz)
    max_hour = int(max_local.hour)

    return (
        min_local.date().isoformat(),
        max_local.date().isoformat(),
        max_local.date().isoformat(),
        max_hour,
    )


def _build_forecast_figure(response: dict[str, Any]) -> go.Figure:
    fig = go.Figure()

    hist_ts, hist_y = _clean_xy(
        response.get("historical_timestamps") or [],
        response.get("historical_values") or [],
    )
    fc_ts, lo, up = _clean_interval(
        response.get("timestamps") or [],
        response.get("lower_bound") or [],
        response.get("upper_bound") or [],
    )
    pred_ts, pred_y = _clean_xy(
        response.get("timestamps") or [],
        response.get("predictions") or [],
    )
    act_ts, act_y = _clean_xy(
        response.get("actual_timestamps") or [],
        response.get("actual_values") or [],
    )

    if hist_ts and hist_y:
        fig.add_trace(
            go.Scatter(
                x=hist_ts,
                y=hist_y,
                mode="lines",
                name="History",
                line=dict(color="#4c6a92", width=2),
            )
        )

    if fc_ts and lo and up:
        fig.add_trace(
            go.Scatter(
                x=fc_ts + fc_ts[::-1],
                y=up + lo[::-1],
                fill="toself",
                fillcolor="rgba(30, 58, 95, 0.20)",
                line=dict(color="rgba(255,255,255,0)"),
                name="90% Prediction Interval",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fc_ts,
                y=lo,
                mode="lines",
                name="Lower",
                line=dict(color="rgba(30, 58, 95, 0.65)", dash="dash", width=1.5),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fc_ts,
                y=up,
                mode="lines",
                name="Upper",
                line=dict(color="rgba(30, 58, 95, 0.65)", dash="dash", width=1.5),
                showlegend=False,
            )
        )

    if pred_ts and pred_y:
        fig.add_trace(
            go.Scatter(
                x=pred_ts,
                y=pred_y,
                mode="lines+markers",
                name="Forecast",
                line=dict(color="#1e3a5f", width=3),
                marker=dict(size=4),
            )
        )

    if act_ts and act_y:
        fig.add_trace(
            go.Scatter(
                x=act_ts,
                y=act_y,
                mode="lines+markers",
                name="Actual",
                line=dict(color="#111111", width=2),
                marker=dict(size=4),
                opacity=0.9,
            )
        )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=50, r=25, t=40, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0.0),
        xaxis_title=f"Timestamp ({DISPLAY_TZ})",
        yaxis_title="Consumption",
        dragmode="zoom",
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.06),
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(count=6, label="6h", step="hour", stepmode="backward"),
                    dict(count=12, label="12h", step="hour", stepmode="backward"),
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(step="all", label="All"),
                ]
            ),
        ),
    )
    return fig


def _build_compare_figure(before: dict[str, Any], after: dict[str, Any]) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Before replay calibration", "After replay calibration"),
    )

    def add_panel(response: dict[str, Any], row: int, interval_fill: str) -> None:
        hist_ts, hist_y = _clean_xy(
            response.get("historical_timestamps") or [],
            response.get("historical_values") or [],
        )
        fc_ts, lo, up = _clean_interval(
            response.get("timestamps") or [],
            response.get("lower_bound") or [],
            response.get("upper_bound") or [],
        )
        pred_ts, pred_y = _clean_xy(
            response.get("timestamps") or [],
            response.get("predictions") or [],
        )
        act_ts, act_y = _clean_xy(
            response.get("actual_timestamps") or [],
            response.get("actual_values") or [],
        )

        if hist_ts and hist_y:
            fig.add_trace(
                go.Scatter(
                    x=hist_ts,
                    y=hist_y,
                    mode="lines",
                    name="History" if row == 1 else "History (after)",
                    line=dict(color="#6b7280", width=1.8),
                    showlegend=(row == 1),
                ),
                row=row,
                col=1,
            )

        if fc_ts and lo and up:
            fig.add_trace(
                go.Scatter(
                    x=fc_ts + fc_ts[::-1],
                    y=up + lo[::-1],
                    fill="toself",
                    fillcolor=interval_fill,
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    name="Prediction interval" if row == 1 else "Prediction interval (after)",
                    showlegend=(row == 1),
                ),
                row=row,
                col=1,
            )

        if pred_ts and pred_y:
            fig.add_trace(
                go.Scatter(
                    x=pred_ts,
                    y=pred_y,
                    mode="lines",
                    name="Forecast" if row == 1 else "Forecast (after)",
                    line=dict(color="#0f4c81", width=3),
                    showlegend=(row == 1),
                ),
                row=row,
                col=1,
            )

        if act_ts and act_y:
            fig.add_trace(
                go.Scatter(
                    x=act_ts,
                    y=act_y,
                    mode="lines",
                    name="Actual" if row == 1 else "Actual (after)",
                    line=dict(color="#d1495b", width=2),
                    showlegend=(row == 1),
                ),
                row=row,
                col=1,
            )

    add_panel(before, row=1, interval_fill="rgba(107, 114, 128, 0.20)")
    add_panel(after, row=2, interval_fill="rgba(15, 76, 129, 0.20)")

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=50, r=25, t=60, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0.0),
        height=820,
    )
    fig.update_xaxes(title_text=f"Timestamp ({DISPLAY_TZ})", row=2, col=1)
    fig.update_yaxes(title_text="Consumption", row=1, col=1)
    fig.update_yaxes(title_text="Consumption", row=2, col=1)
    return fig


def _fmt_pct(v: Any) -> str:
    try:
        if v is None:
            return "n/a"
        return f"{float(v):.1%}"
    except Exception:
        return "n/a"


def _fmt_num(v: Any) -> str:
    try:
        if v is None:
            return "n/a"
        return f"{float(v):.2f}"
    except Exception:
        return "n/a"


def create_dash_app() -> dash.Dash:
    assets_dir = Path(__file__).resolve().parent / "assets"
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        requests_pathname_prefix="/dashboard/",
        suppress_callback_exceptions=True,
        assets_folder=str(assets_dir),
    )

    app.layout = dbc.Container(
        [
            dcc.Store(id="stations-store"),
            dcc.Interval(id="bootstrap-load", interval=800, n_intervals=0, max_intervals=1),
            html.Div(
                [
                    html.H2("Mondrian Forecast Dashboard", className="mb-1"),
                    html.P(
                        "Single anonymized artifact, local CSV source, fixed 96-step horizon.",
                        className="mb-0",
                    ),
                ],
                className="dashboard-header",
            ),
            dbc.Alert(
                "Historical replay / backtesting demo only. Forecasts are generated within the dataset window.",
                color="warning",
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.H5("Station & Inputs", className="section-title"),
                                html.Label("Station"),
                                dcc.Dropdown(id="forecast-station", clearable=False),
                                html.Label(f"Forecast Start ({DISPLAY_TZ})", className="mt-3"),
                                dcc.DatePickerSingle(
                                    id="forecast-start-date",
                                    date=datetime.now().date(),
                                    display_format="YYYY-MM-DD",
                                    className="mb-2",
                                ),
                                dcc.Dropdown(
                                    id="forecast-start-hour",
                                    options=[{"label": f"{h:02d}:00", "value": h} for h in range(0, 24)],
                                    value=0,
                                    placeholder="Hour",
                                ),
                                html.Label("History Hours", className="mt-3"),
                                dbc.Input(
                                    id="forecast-history-hours",
                                    type="number",
                                    value=24,
                                    min=1,
                                    step=1,
                                ),
                                dbc.Button(
                                    "Generate Forecast",
                                    id="run-forecast-btn",
                                    color="primary",
                                    className="w-100 mt-3",
                                ),
                                html.Hr(className="my-4"),
                                html.H5("Replay Calibration", className="section-title"),
                                html.P(
                                    "Calibrate on the trailing N days before the selected replay start, then compare baseline vs recalibrated intervals on that same horizon.",
                                    className="small text-muted",
                                ),
                                html.Label("Calibration Days"),
                                dbc.Input(
                                    id="calibration-days",
                                    type="number",
                                    value=14,
                                    min=1,
                                    step=1,
                                ),
                                dbc.Button(
                                    "Compare Before / After",
                                    id="run-calibration-compare-btn",
                                    color="secondary",
                                    className="w-100 mt-3",
                                ),
                                dbc.Button(
                                    "Reset Runtime Calibration",
                                    id="reset-calibration-btn",
                                    color="outline-secondary",
                                    className="w-100 mt-2",
                                ),
                                html.Div(id="calibration-reset-status", className="mt-2"),
                            ],
                            className="forecast-section",
                        ),
                        md=3,
                        className="forecast-sidebar",
                    ),
                    dbc.Col(
                        [
                            html.Div(id="forecast-status", className="mb-2"),
                            html.Div(id="forecast-coverage", className="text-muted mb-2"),
                            dcc.Graph(id="forecast-graph", style={"height": "620px"}),
                            html.Hr(className="my-4"),
                            html.Div(id="calibration-status", className="mb-2"),
                            html.Div(id="calibration-summary", className="text-muted mb-2"),
                            dcc.Graph(id="calibration-compare-graph", style={"height": "820px"}),
                        ],
                        md=9,
                        className="forecast-main",
                    ),
                ],
                className="forecast-layout g-3",
            ),
        ],
        fluid=True,
    )

    @app.callback(
        Output("stations-store", "data"),
        Output("forecast-station", "options"),
        Output("forecast-station", "value"),
        Output("forecast-start-date", "min_date_allowed"),
        Output("forecast-start-date", "max_date_allowed"),
        Output("forecast-start-date", "date"),
        Output("forecast-start-hour", "value"),
        Output("forecast-history-hours", "min"),
        Output("forecast-history-hours", "value"),
        Input("bootstrap-load", "n_intervals"),
    )
    def load_stations(_n: int):
        stations = _call_api("/stations")
        status_payload = _call_api("/api/mondrian/status")
        options = [{"label": f"{row['name']} ({row['id']})", "value": row["id"]} for row in stations]
        default_station = options[0]["value"] if options else None
        min_date, max_date, default_date, default_hour = _build_demo_scope_notice(status_payload)
        min_history = int(status_payload.get("min_history_hours") or 24)
        default_history = max(24, min_history)
        return (
            stations,
            options,
            default_station,
            min_date,
            max_date,
            default_date,
            default_hour,
            min_history,
            default_history,
        )

    @app.callback(
        Output("forecast-graph", "figure"),
        Output("forecast-status", "children"),
        Output("forecast-coverage", "children"),
        Input("run-forecast-btn", "n_clicks"),
        State("forecast-station", "value"),
        State("forecast-start-date", "date"),
        State("forecast-start-hour", "value"),
        State("forecast-history-hours", "value"),
        prevent_initial_call=True,
    )
    def run_forecast(
        n_clicks: int,
        station: str | None,
        start_date: str | None,
        start_hour: int | None,
        history_hours: int | None,
    ):
        if not n_clicks:
            raise PreventUpdate
        if not station:
            return go.Figure(), dbc.Alert("Please select a station", color="warning"), ""

        payload: dict[str, Any] = {
            "station": station,
            "history_hours": int(history_hours or 24),
            "horizon": 96,
        }
        if start_date:
            try:
                hour_value = int(start_hour) if start_hour is not None else 0
                if hour_value < 0 or hour_value > 23:
                    raise ValueError("hour must be between 0 and 23")
                base_date = datetime.strptime(start_date, "%Y-%m-%d").date()
                local_start = datetime(
                    year=base_date.year,
                    month=base_date.month,
                    day=base_date.day,
                    hour=hour_value,
                    minute=0,
                    second=0,
                    microsecond=0,
                    tzinfo=ZoneInfo(DISPLAY_TZ),
                )
                payload["forecast_start"] = local_start.astimezone(timezone.utc).isoformat()
            except Exception as exc:
                return (
                    go.Figure(),
                    dbc.Alert(f"Invalid forecast start selection: {exc}", color="danger"),
                    "",
                )

        try:
            response = _call_api("/api/mondrian/forecast", method="POST", payload=payload)
        except Exception as exc:
            return go.Figure(), dbc.Alert(f"Forecast request failed: {exc}", color="danger"), ""

        fig = _build_forecast_figure(response)
        width_text = _interval_width_summary(response)
        status_parts = [
            f"Forecast generated for {station}",
            f"horizon={response.get('horizon')}",
            f"start={response.get('forecast_start')}",
        ]
        if width_text:
            status_parts.append(width_text)

        status = dbc.Alert(" | ".join(status_parts), color="success")
        coverage = _coverage_from_forecast(response)
        return fig, status, coverage

    @app.callback(
        Output("calibration-compare-graph", "figure"),
        Output("calibration-status", "children"),
        Output("calibration-summary", "children"),
        Input("run-calibration-compare-btn", "n_clicks"),
        State("forecast-station", "value"),
        State("forecast-start-date", "date"),
        State("forecast-start-hour", "value"),
        State("forecast-history-hours", "value"),
        State("calibration-days", "value"),
        prevent_initial_call=True,
    )
    def run_calibration_compare(
        n_clicks: int,
        station: str | None,
        start_date: str | None,
        start_hour: int | None,
        history_hours: int | None,
        calibration_days: int | None,
    ):
        if not n_clicks:
            raise PreventUpdate
        if not station:
            return go.Figure(), dbc.Alert("Please select a station", color="warning"), ""
        if not start_date:
            return go.Figure(), dbc.Alert("Please select a replay start date", color="warning"), ""

        try:
            hour_value = int(start_hour) if start_hour is not None else 0
            base_date = datetime.strptime(start_date, "%Y-%m-%d").date()
            local_start = datetime(
                year=base_date.year,
                month=base_date.month,
                day=base_date.day,
                hour=hour_value,
                minute=0,
                second=0,
                microsecond=0,
                tzinfo=ZoneInfo(DISPLAY_TZ),
            )
            replay_start = local_start.astimezone(timezone.utc).isoformat()
        except Exception as exc:
            return go.Figure(), dbc.Alert(f"Invalid replay start selection: {exc}", color="danger"), ""

        payload = {
            "station": station,
            "replay_start": replay_start,
            "history_hours": int(history_hours or 24),
            "days": int(calibration_days or 14),
        }

        try:
            response = _call_api(
                "/api/calibration/replay-compare",
                method="POST",
                payload=payload,
                timeout_sec=180,
            )
        except Exception as exc:
            return go.Figure(), dbc.Alert(f"Replay calibration comparison failed: {exc}", color="danger"), ""

        fig = _build_compare_figure(response["before"], response["after"])
        cmp = response.get("comparison") or {}
        before = cmp.get("before") or {}
        after = cmp.get("after") or {}

        status = dbc.Alert(
            " | ".join(
                [
                    f"Replay calibration saved for {station}",
                    f"days={response.get('days')}",
                    f"replay_start={response.get('replay_start')}",
                    f"keys_updated={response.get('calibration_result', {}).get('updated_keys', 'n/a')}",
                ]
            ),
            color="info",
        )
        summary = " | ".join(
            [
                f"Coverage before={_fmt_pct(before.get('coverage'))}",
                f"Coverage after={_fmt_pct(after.get('coverage'))}",
                f"Delta coverage={_fmt_pct(cmp.get('delta_coverage'))}",
                f"Width before={_fmt_num(before.get('mean_interval_width'))}",
                f"Width after={_fmt_num(after.get('mean_interval_width'))}",
                f"Delta width={_fmt_num(cmp.get('delta_mean_interval_width'))}",
            ]
        )
        return fig, status, summary

    @app.callback(
        Output("calibration-reset-status", "children"),
        Input("reset-calibration-btn", "n_clicks"),
        State("forecast-station", "value"),
        prevent_initial_call=True,
    )
    def reset_runtime_calibration(n_clicks: int, station: str | None):
        if not n_clicks:
            raise PreventUpdate
        if not station:
            return dbc.Alert("Select a station before resetting runtime calibration.", color="warning")

        try:
            response = _call_api(
                "/api/calibration/reset",
                method="POST",
                payload={"stations": [station]},
                timeout_sec=30,
            )
        except Exception as exc:
            return dbc.Alert(f"Reset failed: {exc}", color="danger")

        cleared = response.get("stations_cleared") or []
        if station in cleared:
            return dbc.Alert(f"Runtime calibration cleared for {station}.", color="secondary")
        return dbc.Alert(f"No runtime calibration overlay was present for {station}.", color="light")

    return app
