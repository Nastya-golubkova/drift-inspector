#!/usr/bin/env python3
"""Thin wrapper — runner lives under artifacts/assignment1_fit4488/."""
import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "artifacts/assignment1_fit4488/run_fit4488.py"),
    run_name="__main__",
)
