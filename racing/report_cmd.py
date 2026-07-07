"""Validate and serve the static racing report."""

from __future__ import annotations

import argparse
from functools import partial
import http.server
import json
from pathlib import Path
import socketserver
from typing import Any


REQUIRED_FILES = (
    Path("dataset/telemetry.csv"),
    Path("dataset/summary.json"),
    Path("report/assets/data/summary.json"),
    Path("report/assets/data/speed_histograms.json"),
    Path("report/assets/data/speed_profiles.json"),
    Path("report/assets/data/learning_curves.json"),
)
SUMMARY_KEYS = {"experiment", "agents", "per_track"}
EXPERIMENT_KEYS = {"train_steps", "n_eval_tracks", "decel_penalty_coef", "algo", "date"}
AGENT_KEYS = {
    "label",
    "color",
    "mean_lap_time",
    "best_lap_time",
    "mean_speed",
    "std_speed",
    "offtrack_fraction",
    "drift_fraction",
    "hard_brake_events_per_lap",
    "dnf_rate",
}
PER_TRACK_KEYS = {"seed", "lap_time", "mean_speed", "video"}
LEARNING_KEYS = {"steps", "reward", "ep_len"}


def cmd_report(args: argparse.Namespace) -> int:
    """Validate generated report assets and optionally serve them."""

    missing = [str(path) for path in REQUIRED_FILES if not path.exists()]
    if missing:
        print("missing report data; run `python3 -m racing evaluate` first")
        for path in missing:
            print(f"missing {path}")
        return 1
    _validate_contract()
    url = f"http://localhost:{args.port}/report/"
    if args.no_serve:
        print(f"report data valid: {url}")
        return 0
    print(f"serving {url}")
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=".")
    with socketserver.TCPServer(("", int(args.port)), handler) as httpd:
        httpd.serve_forever()
    return 0


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the report subcommand."""

    parser = subparsers.add_parser("report", help="validate and serve the static report")
    parser.add_argument("--no-serve", action="store_true")
    parser.add_argument("--port", type=int, default=8000)


def _validate_contract() -> None:
    summary = _read_json(Path("report/assets/data/summary.json"))
    if set(summary) != SUMMARY_KEYS:
        raise ValueError("summary.json top-level keys do not match report contract")
    if set(summary["experiment"]) != EXPERIMENT_KEYS:
        raise ValueError("summary.json experiment keys do not match report contract")
    if set(summary["agents"]) != {"time", "nobrakes"}:
        raise ValueError("summary.json agents keys do not match report contract")
    for agent in ("time", "nobrakes"):
        if set(summary["agents"][agent]) != AGENT_KEYS:
            raise ValueError(f"summary.json {agent} keys do not match report contract")
    for item in summary["per_track"]:
        if set(item) != PER_TRACK_KEYS:
            raise ValueError("summary.json per_track keys do not match report contract")
        if set(item["lap_time"]) != {"time", "nobrakes"}:
            raise ValueError("summary.json per_track lap_time keys do not match report contract")
        if set(item["mean_speed"]) != {"time", "nobrakes"}:
            raise ValueError("summary.json per_track mean_speed keys do not match report contract")

    histograms = _read_json(Path("report/assets/data/speed_histograms.json"))
    if set(histograms) != {"bin_edges", "time", "nobrakes"}:
        raise ValueError("speed_histograms.json keys do not match report contract")
    profiles = _read_json(Path("report/assets/data/speed_profiles.json"))
    for profile in profiles:
        if set(profile) != {"seed", "s", "time", "nobrakes"}:
            raise ValueError("speed_profiles.json keys do not match report contract")
    learning = _read_json(Path("report/assets/data/learning_curves.json"))
    if set(learning) != {"time", "nobrakes"}:
        raise ValueError("learning_curves.json top-level keys do not match report contract")
    for agent in ("time", "nobrakes"):
        if set(learning[agent]) != LEARNING_KEYS:
            raise ValueError(f"learning_curves.json {agent} keys do not match report contract")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
