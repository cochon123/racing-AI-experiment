"""Animate speed distributions as training checkpoints improve."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from racing.agents import AGENT_SPECS
from racing.env import DEFAULT_DECEL_COEF
from racing.evaluate import AGENT_ORDER, MAX_SPEED, _run_episode, parse_seeds


CHECKPOINT_RE = re.compile(r"(\d+)_steps\.zip$")
FRAME_SIZE = (1100, 620)
EDGES = np.linspace(0.0, MAX_SPEED, 41)


def list_training_snapshots(runs_dir: str | Path) -> list[tuple[int, dict[str, Path]]]:
    """Return sorted (timesteps, {agent: checkpoint_path}) including final models."""

    runs = Path(runs_dir)
    by_step: dict[int, dict[str, Path]] = {}
    for agent in AGENT_ORDER:
        ckpt_dir = runs / agent / "checkpoints"
        if ckpt_dir.is_dir():
            for path in ckpt_dir.glob("*.zip"):
                match = CHECKPOINT_RE.search(path.name)
                if match:
                    step = int(match.group(1))
                    by_step.setdefault(step, {})[agent] = path
    final_step = _final_train_steps(runs)
    for agent in AGENT_ORDER:
        final = runs / agent / "model.zip"
        if final.exists():
            by_step.setdefault(final_step, {})[agent] = final
    return [
        (step, by_step[step])
        for step in sorted(by_step)
        if all(agent in by_step[step] for agent in AGENT_ORDER)
    ]


def _final_train_steps(runs: Path) -> int:
    import pandas as pd

    for agent in AGENT_ORDER:
        progress = runs / agent / "progress.csv"
        if progress.exists():
            df = pd.read_csv(progress)
            if "time/total_timesteps" in df.columns and len(df):
                return int(df["time/total_timesteps"].iloc[-1])
    return 1_511_424


def collect_speed_evolution(
    runs_dir: str = "runs",
    eval_seeds: list[int] | None = None,
    difficulty: float = 0.5,
) -> list[dict[str, Any]]:
    """Roll out every paired checkpoint on the eval track set and histogram speeds."""

    seeds = eval_seeds or list(range(1000, 1016))
    snapshots = list_training_snapshots(runs_dir)
    if not snapshots:
        raise FileNotFoundError(f"no paired checkpoints under {runs_dir}")

    timeline: list[dict[str, Any]] = []
    for step, paths in snapshots:
        print(f"checkpoint {step:,} steps — rolling out {len(seeds)} tracks × 2 agents")
        speeds: dict[str, list[float]] = {agent: [] for agent in AGENT_ORDER}
        for agent in AGENT_ORDER:
            policy = PPO.load(paths[agent])
            for seed in seeds:
                rows, _ = _run_episode(agent, seed, difficulty, policy, decel_coef=DEFAULT_DECEL_COEF)
                speeds[agent].extend(float(r["speed"]) for r in rows)
        frame: dict[str, Any] = {
            "steps": int(step),
            "steps_label": f"{step / 1e6:.2f}M" if step >= 1_000_000 else f"{step / 1e3:.0f}k",
            "histograms": {},
            "mean_speed": {},
        }
        for agent in AGENT_ORDER:
            hist, _ = np.histogram(np.asarray(speeds[agent], dtype=np.float64), bins=EDGES, density=True)
            frame["histograms"][agent] = np.nan_to_num(hist, nan=0.0).round(8).tolist()
            frame["mean_speed"][agent] = float(np.mean(speeds[agent])) if speeds[agent] else 0.0
        timeline.append(frame)
    return timeline


def _smooth_histogram_curve(
    centers: np.ndarray,
    densities: np.ndarray,
    *,
    n_points: int = 200,
    bandwidth: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convolve binned density with a Gaussian kernel for a smooth KDE-like curve."""

    bin_width = float(centers[1] - centers[0]) if len(centers) > 1 else 1.0
    bw = bandwidth or bin_width * 1.5
    xs = np.linspace(float(centers[0]), float(centers[-1]), n_points)
    ys = np.zeros(n_points, dtype=np.float64)
    for center, density in zip(centers, densities, strict=False):
        if density <= 0:
            continue
        ys += density * np.exp(-0.5 * ((xs - center) / bw) ** 2)
    return xs, ys


def render_speed_evolution_video(
    timeline: list[dict[str, Any]],
    out_path: str | Path,
    *,
    fps: int = 30,
    hold_seconds: float = 2.0,
) -> Path:
    """Encode overlapping speed density curves across training checkpoints."""

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    centers = (EDGES[:-1] + EDGES[1:]) / 2.0
    hold_frames = max(1, int(hold_seconds * fps))

    fig, ax = plt.subplots(figsize=(FRAME_SIZE[0] / 100, FRAME_SIZE[1] / 100), dpi=100)
    fig.patch.set_facecolor("#0e1116")
    ax.set_facecolor("#0e1116")

    with imageio.get_writer(
        out,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        ffmpeg_log_level="error",
    ) as writer:
        for frame in timeline:
            ax.clear()
            ax.set_facecolor("#0e1116")
            for agent in AGENT_ORDER:
                color = AGENT_SPECS[agent]["color"]
                densities = np.asarray(frame["histograms"][agent], dtype=np.float64)
                xs, ys = _smooth_histogram_curve(centers, densities)
                ax.plot(xs, ys, color=color, linewidth=2.5, label=AGENT_SPECS[agent]["label"])
                ax.fill_between(xs, ys, color=color, alpha=0.18)
            ax.set_xlim(0, MAX_SPEED)
            ax.set_xlabel("Speed (u/s)", color="#8b949e", fontsize=11)
            ax.set_ylabel("Density", color="#8b949e", fontsize=11)
            ax.tick_params(colors="#8b949e", labelsize=10)
            ax.set_title(
                f"Speed distribution @ {frame['steps']:,} env steps ({frame['steps_label']})",
                color="#e6edf3",
                fontsize=13,
                pad=12,
            )
            for spine in ax.spines.values():
                spine.set_color("#2a2f3a")
            ax.grid(True, color="#2a2f3a", alpha=0.5, linewidth=0.6)
            legend = ax.legend(loc="upper right", frameon=False, fontsize=10)
            for text in legend.get_texts():
                text.set_color("#e6edf3")
            mean_bits = " · ".join(
                f"{AGENT_SPECS[a]['label']}: {frame['mean_speed'][a]:.1f} u/s mean"
                for a in AGENT_ORDER
            )
            ax.text(0.02, 0.97, mean_bits, transform=ax.transAxes, color="#8b949e", fontsize=9, va="top")

            fig.tight_layout()
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            rgb = rgba[:, :, :3].copy()
            for _ in range(hold_frames):
                writer.append_data(rgb)

    plt.close(fig)
    return out


def build_evolution(
    runs_dir: str = "runs",
    out_video: str = "report/assets/videos/speed_distribution_evolution.mp4",
    out_json: str = "report/assets/data/speed_evolution.json",
    eval_seeds: str = "1000-1015",
    difficulty: float = 0.5,
    hold_seconds: float = 2.0,
) -> Path:
    """Collect checkpoint histograms and render the evolution MP4."""

    seeds = parse_seeds(eval_seeds)
    timeline = collect_speed_evolution(runs_dir, seeds, difficulty)
    json_path = Path(out_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps({"bin_edges": EDGES.round(6).tolist(), "frames": timeline}, indent=2) + "\n",
        encoding="utf-8",
    )
    video_path = render_speed_evolution_video(timeline, out_video, hold_seconds=hold_seconds)
    print(f"wrote {video_path} ({len(timeline)} checkpoints, {len(seeds)} tracks each)")
    print(f"wrote {json_path}")
    return video_path


def cmd_evolution_video(args: argparse.Namespace) -> int:
    build_evolution(
        runs_dir=args.runs_dir,
        out_video=args.out,
        out_json=args.json,
        eval_seeds=args.seeds,
        difficulty=args.difficulty,
        hold_seconds=args.hold,
    )
    return 0


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "evolution-video",
        help="render speed-distribution histogram video across training checkpoints",
    )
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out", default="report/assets/videos/speed_distribution_evolution.mp4")
    parser.add_argument("--json", default="report/assets/data/speed_evolution.json")
    parser.add_argument("--seeds", default="1000-1015")
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--hold", type=float, default=2.0, help="seconds to hold each checkpoint frame")
