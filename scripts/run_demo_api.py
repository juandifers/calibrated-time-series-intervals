#!/usr/bin/env python3
"""Run the project2 FastAPI + Dash demo service."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run project2 demo API service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project2_root = Path(__file__).resolve().parents[1]
    if str(project2_root) not in sys.path:
        sys.path.insert(0, str(project2_root))
    os.chdir(project2_root)
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
