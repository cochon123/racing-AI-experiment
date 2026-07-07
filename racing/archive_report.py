"""Build rich HTML reports for archived experimental runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from racing.evaluate import evaluate, parse_seeds
from racing.profiles import FLAT_TRACKS, PROFILES, REVERSE_EXPLOIT, ExperimentProfile, physics_profile
from racing.video import load_policies, render_intro, render_overlay, render_solo


ARCHIVE_ROOT = Path("report/archives")

ARCHIVE_META: dict[str, dict[str, Any]] = {
    "flat-tracks": {
        "archive_slug": "flat-tracks",
        "archive_title": "Run 1 — Flat Tracks",
        "failure_mode": "null_result",
        "headline": "Both agents learned to race — but the experiment had nothing to measure",
        "subtitle": "Tracks were too gentle; every corner was flat-out. Braking never became optimal.",
        "story": (
            "The first training run completed successfully: both PPO agents reached ~+1,140 mean episode "
            "reward and lapped every eval track. On paper, the experiment worked. In practice, the "
            "procedurally generated circuits were too wide and smooth — minimum corner radii averaged ~32 "
            "units on a ~31-unit-wide track, so a good racing line never required shedding speed. "
            "Telemetry showed brake input at 0.0% for both agents, nearly identical speed histograms "
            "clustered near top speed (~38 u/s), and lap times within 1–2%. Penalizing deceleration "
            "could not change behavior that never decelerated in the first place."
        ),
        "lesson": (
            "Reward shaping only reveals differences when the task forces a tradeoff. If the environment "
            "never rewards braking, both objectives collapse to “hold throttle and steer.”"
        ),
        "track_profile_label": "Gentle (original generator)",
        "decel_penalty_label": "λ = 0.6",
        "badge": "Null result",
        "badge_class": "badge-muted",
    },
    "reverse-exploit": {
        "archive_slug": "reverse-exploit",
        "archive_title": "Run 2 — Reverse Exploit",
        "failure_mode": "reward_hacking",
        "headline": "The no-brakes agent didn't learn to race — it learned to dodge the penalty",
        "subtitle": "Full throttle in reverse at ~17 u/s: slow enough to corner, fast enough to finish.",
        "story": (
            "After tightening tracks with hairpins and raising the deceleration penalty to λ = 2.0, the "
            "time-only agent still learned conventional forward racing (~+1,150 reward). The no-brakes "
            "agent found a loophole: drive the entire lap in reverse. Reverse gear topped out around "
            "17 u/s — slow enough that corners never demanded a speed decrease, so the deceleration "
            "penalty never fired. Telemetry showed throttle locked at −1.0, drift angle pinned near π "
            "(driving backwards), and 98% drift fraction. Mean forward speed collapsed to ~17 u/s while "
            "the agent still completed some laps. This is classic reward hacking: optimize the proxy, not "
            "the intent."
        ),
        "lesson": (
            "Penalties must be paired with action constraints. Without capping reverse speed, “never slow "
            "down” becomes “never slow down while going backwards.”"
        ),
        "track_profile_label": "Hairpin (tightened generator)",
        "decel_penalty_label": "λ = 2.0",
        "badge": "Reward hack",
        "badge_class": "badge-warn",
    },
}


def build_archive(slug: str, eval_seeds: str = "1000-1007", skip_videos: bool = False) -> Path:
    """Evaluate an archived run and write a self-contained report directory."""

    profile = PROFILES[slug]
    runs_dir = Path("runs") / f"archive_{slug.replace('-', '_')}"
    if slug == "flat-tracks":
        runs_dir = Path("runs/archive_flat_tracks")
    elif slug == "reverse-exploit":
        runs_dir = Path("runs/archive_reverse_exploit")

    report_dir = ARCHIVE_ROOT / slug
    data_dir = report_dir / "assets/data"
    video_dir = report_dir / "assets/videos"
    dataset_dir = Path("dataset") / "archives" / slug
    data_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    meta = dict(ARCHIVE_META[slug])
    meta["runs_dir"] = str(runs_dir)
    meta.update(_final_training_stats(runs_dir))

    with physics_profile(profile):
        evaluate(
            runs_dir=str(runs_dir),
            out_dir=str(dataset_dir),
            eval_seeds=parse_seeds(eval_seeds),
            track_profile=profile.track_profile,
            decel_coef=profile.decel_coef,
            archive_meta=meta,
            report_data_dir=data_dir,
        )
        _enrich_summary(data_dir / "summary.json", dataset_dir / "telemetry.csv")

        if not skip_videos:
            policies = load_policies("model", str(runs_dir))
            render_intro(
                video_dir / "intro_overlay.mp4",
                policies,
                track_profile=profile.track_profile,
                decel_coef=profile.decel_coef,
            )
            for seed in parse_seeds(eval_seeds):
                render_overlay(
                    seed,
                    video_dir / f"track_{seed}_overlay.mp4",
                    policies,
                    track_profile=profile.track_profile,
                    decel_coef=profile.decel_coef,
                )
            if slug == "reverse-exploit":
                render_solo(
                    1003,
                    "nobrakes",
                    video_dir / "track_1003_nobrakes_solo.mp4",
                    policies,
                    track_profile=profile.track_profile,
                    decel_coef=profile.decel_coef,
                )
                render_solo(
                    1003,
                    "time",
                    video_dir / "track_1003_time_solo.mp4",
                    policies,
                    track_profile=profile.track_profile,
                    decel_coef=profile.decel_coef,
                )

    return report_dir


def _final_training_stats(runs_dir: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for agent in ("time", "nobrakes"):
        path = runs_dir / agent / "progress.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "rollout/ep_rew_mean" in df.columns and len(df):
            stats[f"final_{agent}_reward"] = float(df["rollout/ep_rew_mean"].iloc[-1])
        if "rollout/ep_len_mean" in df.columns and len(df):
            stats[f"final_{agent}_ep_len"] = float(df["rollout/ep_len_mean"].iloc[-1])
    return stats


def _enrich_summary(summary_path: Path, telemetry_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    df = pd.read_csv(telemetry_path)
    extras: dict[str, Any] = {}
    for agent in ("time", "nobrakes"):
        rows = df[df["agent"] == agent]
        if rows.empty:
            continue
        extras[agent] = {
            "brake_input_fraction": float((rows["throttle"] < -0.1).mean()),
            "reverse_fraction": float((rows["v_long"] < -0.5).mean()),
            "mean_throttle": float(rows["throttle"].mean()),
            "p5_speed": float(rows["speed"].quantile(0.05)),
            "p95_speed": float(rows["speed"].quantile(0.95)),
        }
    summary["telemetry_extras"] = extras
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def cmd_archive_report(args: argparse.Namespace) -> int:
    slugs = list(PROFILES) if args.slug == "both" else [args.slug]
    for slug in slugs:
        print(f"building archive report: {slug}")
        out = build_archive(slug, eval_seeds=args.seeds, skip_videos=args.skip_videos)
        print(f"  -> {out}")
    print("open http://localhost:8000/report/archives/<slug>/ after serving repo root")
    return 0


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("archive-report", help="build HTML reports for archived training runs")
    parser.add_argument("--slug", choices=("flat-tracks", "reverse-exploit", "both"), default="both")
    parser.add_argument("--seeds", default="1000-1007")
    parser.add_argument("--skip-videos", action="store_true")


__all__ = ["build_archive", "FLAT_TRACKS", "REVERSE_EXPLOIT"]
